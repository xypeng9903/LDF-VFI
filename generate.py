import argparse
import logging
import os
import json
from pathlib import Path
import re
from functools import partial
from typing import Iterable, List, Optional, Sequence, Union
from PIL import Image
import math

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import torchvision
import diffusers
from torchcodec.decoders import VideoDecoder
from einops import rearrange, repeat

from training.models import WanTransformer3DModel
from training.distributed import parallel_state_sp as sp_state
from training.util import print_module_summary
from training.data.stsr_dataset import upsample_temporal
from training.models.precond import (
    Precond,
    Wan2_1SpatialTiledEncoder3D,
    Wan2_1SpatialTiledConditionEncoder3D,
    Wan2_1SpatialTiledConditionEncoder3Dv2,
    MaskSpatialTiledEncoder3D,
    SpatialTiledEncoder3D,
    SpatialTiledConditionEncoder3D
)

logger = get_logger(__name__, log_level="INFO")

#----------------------------------------------------------------------------
# Helpers.

def _natural_key(s: str):
    # Split into alpha and numeric chunks for natural sort, e.g., 1, 2, 10
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]


class VideoFrameLazyReader(Sequence[torch.Tensor]):
    """
    A lazy, minimal-memory reader over a folder of video frames that indexes like a THWC uint8 tensor.

    - reader[i] -> torch.uint8 tensor of shape (H, W, C)
    - reader[start:end:stride] -> torch.uint8 tensor of shape (T, H, W, C)
    - len(reader) -> number of frames (T)
    - reader.shape -> (T, H, W, C)  (H, W, C inferred from first frame; channels standardized to 3 via RGB)

    Notes:
    - Images are loaded on-demand using PIL and converted to RGB (3 channels).
    - Minimal memory: only requested frames are loaded; no frame caching by default.
    - Negative indices and reversed slices are supported.
    - Empty slices return shape (0, H, W, C).
    """

    def __init__(self, dir_path: str, extensions: Optional[Iterable[str]] = None, dimension_order="NHWC"):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if extensions is None:
            extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        if not files:
            raise ValueError(f"No image frames found in: {dir_path}")

        # Natural sort, important for 0001.png, 0002.png, ...
        files.sort(key=lambda p: _natural_key(os.path.basename(p)))
        self._files: List[str] = files
        self._T: int = len(files)

        # Infer (H, W, C) from the first frame; standardize to RGB (C=3)
        with Image.open(self._files[0]) as im0:
            im0 = im0.convert("RGB")
            w, h = im0.size
        self._H, self._W, self._C = h, w, 3

        assert dimension_order in ["NHWC", "NCHW"]
        self._dim_order = dimension_order

    def __len__(self) -> int:
        return self._T

    def _load_frame(self, idx: int) -> torch.Tensor:
        # Bounds and negative index normalization
        if idx < 0:
            idx += self._T
        if idx < 0 or idx >= self._T:
            raise IndexError(f"Frame index out of range: {idx}")

        path = self._files[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")  # enforce 3 channels
            # Convert to torch.uint8 HWC without keeping extra copies
            # PIL -> bytes -> torch is efficient enough for single frame
            arr = torch.frombuffer(im.tobytes(), dtype=torch.uint8)
            # Reshape to HWC
            frame = arr.view(self._H, self._W, self._C)
            if self._dim_order == "NCHW":
                frame = frame.permute(2, 0, 1)
            return frame.clone()

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> torch.Tensor:
        if isinstance(idx, int):
            return self._load_frame(idx)

        if isinstance(idx, slice):
            indices = list(range(*idx.indices(self._T)))
            return self._gather_indices(indices)

        # Fancy indexing: list/tuple/torch.Tensor/numpy array of indices
        if isinstance(idx, (list, tuple)):
            return self._gather_indices(list(idx))

        # Optional: support torch.Tensor or numpy arrays of indices
        try:
            import numpy as np  # type: ignore
            if isinstance(idx, np.ndarray):
                return self._gather_indices(idx.tolist())
        except Exception:
            pass

        if torch.is_tensor(idx):
            return self._gather_indices(idx.cpu().tolist())

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def _gather_indices(self, indices: List[int]) -> torch.Tensor:
        n = len(indices)
        if n == 0:
            # Return an empty tensor with the right THWC shape
            return torch.empty((0, self._H, self._W, self._C), dtype=torch.uint8)

        # Normalize negatives and range-check once
        norm_idx = []
        for i in indices:
            ii = i + self._T if i < 0 else i
            if ii < 0 or ii >= self._T:
                raise IndexError(f"Frame index out of range: {i}")
            norm_idx.append(ii)

        # Preallocate and stream-load
        if self._dim_order == "NHWC":
            out = torch.empty((n, self._H, self._W, self._C), dtype=torch.uint8)
        elif self._dim_order == "NCHW":
            out = torch.empty((n, self._C, self._H, self._W), dtype=torch.uint8)
        for k, ii in enumerate(norm_idx):
            out[k] = self._load_frame(ii)
        return out


def vae_decode(vae, xt, **kwargs):
    if isinstance(vae, SpatialTiledEncoder3D):
        pred = vae.decode(xt)
    elif isinstance(vae, SpatialTiledConditionEncoder3D):
        lq = kwargs.get('lq')
        msk = kwargs.get('msk')
        decode_H = xt.shape[-2] * vae.spatial_compression_ratio
        decode_W = xt.shape[-1] * vae.spatial_compression_ratio
        lq = F.pad(lq, (0, decode_W - lq.shape[-1], 0, decode_H - lq.shape[-2]))
        lq = rearrange(lq, 'b c (nt t) h w -> b nt c t h w', nt=xt.shape[1])
        msk = msk[..., 0, 0]
        msk = repeat(msk, 'b c t -> b c t h w', h=decode_H, w=decode_W)
        msk = rearrange(msk, 'b c (nt t) h w -> b nt c t h w', nt=xt.shape[1])
        pred = vae.decode(xt, lq, msk)
    else:
        raise RuntimeError("Unsupported VAE type.")
    return pred

#----------------------------------------------------------------------------
# Samplers.

def sample_euler(
    lq_reader,  # TCHW, uint8 tensor
    model,
    train_num_frames=60, temporal_upsample="nearest",  # train hparams
    spatial_sf=1, temporal_sf=4, num_steps=10, t_shift=1.0,  # sampling hparams
    device='cuda', verbose=True,
):        
    # assert len(lq_reader) == train_num_frames // temporal_sf
    assert spatial_sf == 1
    
    lq_H, lq_W = lq_reader[0].shape[-2], lq_reader[0].shape[-1]
    input_T = train_num_frames
    input_H = lq_H * spatial_sf
    input_W = lq_W * spatial_sf
        
    ts = torch.linspace(1.0, 0.0, steps=num_steps + 1).numpy()
    ts = t_shift * ts / (1 + (t_shift - 1) * ts)
            
    # Get lq and msk latents.
    lq = lq_reader[:]  # TCHW uint8
    msk = torch.zeros(input_T, dtype=torch.bool, device=device)
    msk[::temporal_sf] = True    
    lq = upsample_temporal(lq, msk, mode=temporal_upsample)  
    lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
    msk = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=torch.bfloat16)
    y = model.lq_encoder.encode(lq, for_train=True)
    m = model.msk_encoder.encode(msk, for_train=True)
    
    # Flow sampling via ODE integration.
    xt = torch.randn_like(model.vae.encode(lq, for_train=True))
    pbar = tqdm(range(num_steps), disable=not verbose, desc="Sampling")
    for i in pbar:
        t = torch.tensor([ts[i]]).to(xt).expand(xt.shape[:-4]).contiguous()
        vt = model.predict_v(xt, t, y, m)
        xt = xt + vt * (ts[i + 1] - ts[i])
    xt = rearrange(xt, 'b nt nh nw c t h w -> b nt c t (nh h) (nw w)')
    
    if isinstance(model.vae, SpatialTiledEncoder3D):
        pred = model.vae.decode(xt)
    elif isinstance(model.vae, SpatialTiledConditionEncoder3D):
        decode_H = xt.shape[-2] * model.vae.spatial_compression_ratio
        decode_W = xt.shape[-1] * model.vae.spatial_compression_ratio
        lq = F.pad(lq, (0, decode_W - lq.shape[-1], 0, decode_H - lq.shape[-2]))
        lq = rearrange(lq, 'b c (nt t) h w -> b nt c t h w', nt=xt.shape[1])
        msk = msk[..., 0, 0]
        msk = repeat(msk, 'b c t -> b c t h w', h=decode_H, w=decode_W)
        msk = rearrange(msk, 'b c (nt t) h w -> b nt c t h w', nt=xt.shape[1])
        pred = model.vae.decode(xt, lq, msk)
    
    # Convert to TCHW uint8 and crop to input size.
    pred = pred[..., :input_H, :input_W]
    pred_uint8 = rearrange(pred, '1 c t h w -> t c h w').add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8)
    return pred_uint8  # TCHW, uint8
            

def sample_causal(
    lq_reader, # TCHW, uint8 tensor
    model: Precond, train_num_frames=60, tile_min_t=20, temporal_upsample="nearest",
    spatial_sf=1, temporal_sf=4, nT_cond=1, num_steps=10, t_cond=0.1, t_shift=1.0, # sampling hparams
    device='cuda', weight_dtype=torch.bfloat16, verbose=True, pbar_name: str | None = None,
):
    """Causal autoregressive sampling (middle-tile only).

    与 sample_skip_concat 对齐：每个窗口只保留中间的 latent tiles [nT_cond : nT - nT_cond]，
    两侧边界各 nT_cond tiles 只用于上下文与自回归传递，不直接解码输出。

    Output per chunk = (nT - 2 * nT_cond) * tile_min_t frames (except tail which may be shorter).

    Yields:
        (lq_uint8_chunk, pred_uint8_chunk) both TCHW uint8 for the middle region.
    """
    H, W = lq_reader[0].shape[1], lq_reader[0].shape[2]
    nT = train_num_frames // tile_min_t
    assert spatial_sf == 1, "Current implementation assumes spatial_sf == 1"
    # assert nT_cond < nT // 2, "nT_cond must be < nT/2 to have a middle region"

    # Time schedule.
    ts = torch.linspace(1.0, 0.0, steps=num_steps + 1).numpy()
    ts = t_shift * ts / (1 + (t_shift - 1) * ts)

    # Boolean mask over HR timeline where LQ exists (after temporal upsample).
    total_msk = torch.zeros(len(lq_reader) * temporal_sf, dtype=torch.bool)
    total_msk[::temporal_sf] = True

    # Frames produced per causal chunk (middle slice only).
    stride_out = (nT - 2 * nT_cond) * tile_min_t

    # ------------------------------------------
    # First chunk: sample full window; decode middle part.
    
    input_start = 0
    input_end = train_num_frames
    input_lq_start = 0
    input_lq_end = total_msk[:input_end].sum()

    # Middle region time indices (HR) within the window.
    mid_start = 0
    mid_end = (nT - nT_cond) * tile_min_t
    output_lq_start = total_msk[:mid_start].sum()
    output_lq_end = total_msk[:mid_end].sum()
    output_lq = lq_reader[output_lq_start:output_lq_end]

    msk = total_msk[input_start:input_end]
    msk = F.pad(msk, (0, train_num_frames - len(msk)))
    lq = lq_reader[input_lq_start:input_lq_end]
    lq = upsample_temporal(lq, msk, mode=temporal_upsample)
    lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
    msk_t = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)

    y = model.lq_encoder.encode(lq, for_train=True)
    m = model.msk_encoder.encode(msk_t, for_train=True)
    xt = torch.randn_like(y)

    desc = pbar_name or f"Causal {output_lq_start}-{output_lq_end} / {len(lq_reader)} (first chunk)"
    pbar = tqdm(range(num_steps), disable=not verbose, desc=desc)
    for i in pbar:
        t = torch.tensor([ts[i]], device=device, dtype=weight_dtype).expand(xt.shape[:-4]).contiguous()
        vt = model.predict_v(xt, t, y, m)
        xt = xt + vt * (ts[i + 1] - ts[i])

    # Slice middle region, then save AR state as the last nT_cond of the sliced middle (aligns with skip_concat).
    xt_mid = xt[:, : nT - nT_cond]
    x0_prev = xt_mid[:, -nT_cond:]

    # Decode middle.
    xt_dec = rearrange(xt_mid, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
    lq_dec = lq[:, :, mid_start:mid_end]
    msk_dec = msk_t[:, :, mid_start:mid_end]
    pred = vae_decode(model.vae, xt_dec, lq=lq_dec, msk=msk_dec)[..., :H, :W]
    output_pred = rearrange(pred, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()
    yield output_lq, output_pred

    produced_t = mid_end  # global HR frame index (exclusive) of produced output so far

    # ------------------------------------------
    # Subsequent causal chunks.
    while produced_t < len(total_msk):
        # Determine tail.
        remaining = len(total_msk) - produced_t
        is_last = remaining <= stride_out

        # Window covers past boundary + prospective middle + future boundary (future boundary is noise, not decoded).
        window_start = produced_t - nT_cond * tile_min_t
        window_end = window_start + train_num_frames
        window_end = min(window_end, len(total_msk))

        input_lq_start = total_msk[:window_start].sum()
        input_lq_end = total_msk[:window_end].sum()

        # Middle region indices for this window.
        mid_start = produced_t
        mid_end = min(produced_t + stride_out, len(total_msk)) if not is_last else len(total_msk)
        output_lq_start = total_msk[:mid_start].sum()
        output_lq_end = total_msk[:mid_end].sum()
        output_lq = lq_reader[output_lq_start:output_lq_end]

        # Mask & LQ prep.
        msk = total_msk[window_start:window_end]
        msk = F.pad(msk, (0, train_num_frames - len(msk)))
        lq = lq_reader[input_lq_start:input_lq_end]
        lq = upsample_temporal(lq, msk, mode=temporal_upsample)
        lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
        msk_t = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)

        y = model.lq_encoder.encode(lq, for_train=True)
        m = model.msk_encoder.encode(msk_t, for_train=True)

        # Latent assembly: left boundary fixed (x0_prev), middle + right boundary (noise) combined in xt_cat.
        xt_prev = x0_prev * (1 - t_cond) + torch.randn_like(x0_prev) * t_cond
        # We need (nT - nT_cond) tiles after concatenation with xt_prev to make full nT.
        xt_cat = torch.randn_like(y[:, nT_cond:])  # shape (1, nT - nT_cond, ...)

        desc_mid = pbar_name or f"Causal {output_lq_start}-{output_lq_end} / {len(lq_reader)}" + (" (last chunk)" if is_last else "")
        pbar = tqdm(range(num_steps), disable=not verbose, desc=desc_mid)
        for i in pbar:
            xt = torch.cat([xt_prev, xt_cat], dim=1)  # full window (nT tiles)
            t_prev = torch.tensor([t_cond], device=device, dtype=weight_dtype).expand(xt_prev.shape[:-4])
            t_cat = torch.tensor([ts[i]], device=device, dtype=weight_dtype).expand(xt_cat.shape[:-4])
            t = torch.cat([t_prev, t_cat], dim=1)
            vt = model.predict_v(xt, t, y, m)[:, nT_cond:]  # gradient only for cat part
            xt_cat = xt_cat + vt * (ts[i + 1] - ts[i])

        # Middle region to decode and AR state from its tail (slice后最后 nT_cond 块)
        xt_mid = xt_cat[:, : nT - 2 * nT_cond]
        x0_prev = xt_mid[:, -nT_cond:]

        # Decode middle segment (always decode full middle region; crop only after decode for tail).
        mid_lq_start = nT_cond * tile_min_t
        mid_lq_end = (nT - nT_cond) * tile_min_t
        lq_dec = lq[:, :, mid_lq_start:mid_lq_end]
        msk_dec = msk_t[:, :, mid_lq_start:mid_lq_end]

        xt_dec = rearrange(xt_mid, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
        pred = vae_decode(model.vae, xt_dec, lq=lq_dec, msk=msk_dec)[..., :H, :W]
        pred_uint8 = rearrange(pred, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()
        # If tail, crop decoded frames to actual remaining frames.
        frames_remaining = mid_end - mid_start
        if frames_remaining < stride_out:
            pred_uint8 = pred_uint8[:frames_remaining]
        yield output_lq, pred_uint8

        produced_t = mid_end


def sample_skip_concat(
    lq_reader, # TCHW, uint8 tensor
    model: Precond, train_num_frames=60, tile_min_t=20, temporal_upsample="nearest",
    spatial_sf=1, temporal_sf=4, nT_cond=1, num_steps=10, t_cond=0.1, t_shift=1.0, # sampling hparams
    device='cuda', weight_dtype=torch.bfloat16, verbose=True, pbar_name: str | None = None,
):  
    H, W = lq_reader[0].shape[1], lq_reader[0].shape[2]
    nT = train_num_frames // tile_min_t
    ts = torch.linspace(1.0, 0.0, steps=num_steps + 1).numpy()
    ts = t_shift * ts / (1 + (t_shift - 1) * ts)
    total_msk = torch.zeros(len(lq_reader) * temporal_sf, dtype=torch.bool)
    total_msk[::temporal_sf] = True

    # ------------------------------------------
    # The first chunk.
    
    # Inputs.
    input_start = 0
    input_end = train_num_frames
    output_start = 0
    output_end = (nT - nT_cond) * tile_min_t

    input_lq_start = 0
    input_lq_end = total_msk[ : input_end].sum()

    output_lq_start = 0
    output_lq_end = total_msk[ : output_end].sum()

    msk = total_msk[input_start : input_end]
    lq = lq_reader[input_lq_start : input_lq_end]
    output_lq = lq_reader[output_lq_start : output_lq_end]

    msk = F.pad(msk, (0, train_num_frames - len(msk)))
    lq = upsample_temporal(lq, msk, mode=temporal_upsample) 
    lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
    msk = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)
    y = model.lq_encoder.encode(lq, for_train=True)
    m = model.msk_encoder.encode(msk, for_train=True)
    xt = torch.randn_like(y)
    
    # Sampling.
    pbar = tqdm(range(num_steps), disable=not verbose, desc=f"Upscaling {output_lq_start}-{output_lq_end} / {len(lq_reader)} (first chunk)")
    for i in pbar:
        t = torch.tensor([ts[i]], device=device, dtype=weight_dtype).expand(xt.shape[:-4]).contiguous()
        vt = model.predict_v(xt, t, y, m)
        xt = xt + vt * (ts[i + 1] - ts[i])
    xt = xt[:,  : nT - nT_cond]
    
    # Save for AR.
    x0_prev = xt[:, xt.shape[1] - nT_cond : ]
    
    # VAE decode.
    xt = rearrange(xt, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
    lq = lq[:, :, :(nT - nT_cond) * tile_min_t]
    msk = msk[:, :, :(nT - nT_cond) * tile_min_t]
    pred = vae_decode(model.vae, xt, lq=lq, msk=msk)[..., :H, :W]   
    output_pred = rearrange(pred, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()
    
    last_end_idx = output_end
    yield output_lq, output_pred

    t0 = (nT - nT_cond) * tile_min_t
    t0_stride = (nT - nT_cond * 2) * tile_min_t
    while True:
        if t0 + t0_stride > len(total_msk) - 1:
            break

        # ------------------------------------------
        # The skip chunk.

        input_start = t0 + t0_stride - nT_cond * tile_min_t
        input_end = t0 + t0_stride * 2 + nT_cond * tile_min_t
        
        output_start = t0 + t0_stride
        output_end = t0 + t0_stride * 2

        input_lq_start = total_msk[ : input_start].sum()
        input_lq_end = total_msk[ : input_end].sum()

        output_lq_start = total_msk[ : output_start].sum()
        output_lq_end = total_msk[ : output_end].sum()

        last_end_idx = output_end

        msk = total_msk[input_start : input_end]
        lq = lq_reader[input_lq_start : input_lq_end]
        output_lq_skip = lq_reader[output_lq_start : output_lq_end]

        msk = F.pad(msk, (0, train_num_frames - len(msk)))
        lq = upsample_temporal(lq, msk, mode=temporal_upsample)  
        lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
        msk = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)
        y = model.lq_encoder.encode(lq, for_train=True)
        m = model.msk_encoder.encode(msk, for_train=True)
        xt = torch.randn_like(y)

        # Sampling.
        pbar = tqdm(range(num_steps), disable=not verbose, desc=f"Upscaling {output_lq_start}-{output_lq_end} / {len(lq_reader)} (skip chunk)")
        for i in pbar:
            t = torch.tensor([ts[i]], device=device, dtype=weight_dtype).expand(xt.shape[:-4]).contiguous()
            vt = model.predict_v(xt, t, y, m)
            xt = xt + vt * (ts[i + 1] - ts[i])
        xt = xt[:, nT_cond : nT - nT_cond]

        # Save for AR.
        x0_skip = xt[:, : nT_cond] 
        x0_prev_ = xt[:, xt.shape[1] - nT_cond : ]    
        
        # VAE decode.
        xt = rearrange(xt, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
        lq = lq[:, :, nT_cond * tile_min_t : (nT - nT_cond) * tile_min_t]
        msk = msk[:, :, nT_cond * tile_min_t : (nT - nT_cond) * tile_min_t]
        pred = vae_decode(model.vae, xt, lq=lq, msk=msk)[..., :H, :W]   
        output_pred_skip = rearrange(pred, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()

        # ------------------------------------------
        # The concatenate chunk.

        input_start = t0 - nT_cond * tile_min_t
        input_end = t0 + t0_stride + nT_cond * tile_min_t
        
        output_start = t0
        output_end = t0 + t0_stride

        input_lq_start = total_msk[ : input_start].sum()
        input_lq_end = total_msk[ : input_end].sum()

        output_lq_start = total_msk[ : output_start].sum()
        output_lq_end = total_msk[ : output_end].sum()

        msk = total_msk[input_start : input_end]
        lq = lq_reader[input_lq_start : input_lq_end]
        output_lq_cat = lq_reader[output_lq_start : output_lq_end]

        msk = F.pad(msk, (0, train_num_frames - len(msk)))
        lq = upsample_temporal(lq, msk, mode=temporal_upsample)  
        lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
        msk = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)
        y = model.lq_encoder.encode(lq, for_train=True)
        m = model.msk_encoder.encode(msk, for_train=True)
        
        xt_cat = torch.randn_like(y[:, nT_cond : nT - nT_cond])
        xt_prev = x0_prev * (1 - t_cond) + torch.randn_like(x0_prev) * t_cond
        xt_skip = x0_skip * (1 - t_cond) + torch.randn_like(x0_skip) * t_cond
        t_prev = torch.tensor([t_cond]).expand(x0_prev.shape[:-4])
        t_skip = torch.tensor([t_cond]).expand(x0_skip.shape[:-4])
        xt = torch.cat([xt_prev, xt_cat, xt_skip], dim=1)

        # Sampling.
        pbar = tqdm(range(num_steps), disable=not verbose, desc=f"Upscaling {output_lq_start}-{output_lq_end} / {len(lq_reader)} (concat chunk)")
        for i in pbar:
            xt = torch.cat([xt_prev, xt_cat, xt_skip], dim=1)
            t_cat = torch.tensor([ts[i]]).expand(xt_cat.shape[:-4]).contiguous()
            t = torch.cat([t_prev, t_cat, t_skip], dim=1).to(device=device, dtype=weight_dtype)
            vt = model.predict_v(xt, t, y, m)[:, nT_cond : nT - nT_cond]
            xt_cat = xt_cat + vt * (ts[i + 1] - ts[i])
        xt = xt_cat

        # VAE decode.
        xt = rearrange(xt, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
        lq = lq[:, :, nT_cond * tile_min_t : (nT - nT_cond) * tile_min_t]
        msk = msk[:, :, nT_cond * tile_min_t : (nT - nT_cond) * tile_min_t]
        pred = vae_decode(model.vae, xt, lq=lq, msk=msk)[..., :H, :W] 
        output_pred_cat = rearrange(pred, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()

        yield output_lq_cat, output_pred_cat
        yield output_lq_skip, output_pred_skip

        x0_prev = x0_prev_
        t0 += t0_stride * 2

    # ------------------------------------------
    # The last chunk.

    if t0 < len(total_msk):
        input_start = t0 - nT_cond * tile_min_t
        input_lq_start = total_msk[:input_start].sum()
        output_lq_start = total_msk[:t0].sum()
        
        msk = total_msk[input_start:]
        lq = lq_reader[input_lq_start:]
        output_lq_tail = lq_reader[output_lq_start:]
        
        msk = F.pad(msk, (0, train_num_frames - len(msk)))
        lq = upsample_temporal(lq, msk, mode=temporal_upsample)
        lq = rearrange(lq, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype).div(127.5).sub(1)
        msk_t = repeat(msk, 't -> 1 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)
        y = model.lq_encoder.encode(lq, for_train=True)
        m = model.msk_encoder.encode(msk_t, for_train=True)

        xt_prev = x0_prev * (1 - t_cond) + torch.randn_like(x0_prev) * t_cond
        xt_cat = torch.randn_like(y[:, nT_cond:])

        pbar = tqdm(range(num_steps), disable=not verbose, desc=f"Upscaling {output_lq_start}-{len(lq_reader)} / {len(lq_reader)} (last chunk)")
        for i in pbar:
            xt = torch.cat([xt_prev, xt_cat], dim=1)
            t_prev = torch.tensor([t_cond]).expand(xt_prev.shape[:-4])
            t_cat = torch.tensor([ts[i]]).expand(xt_cat.shape[:-4])
            t = torch.cat([t_prev, t_cat], dim=1).to(device=device, dtype=weight_dtype)
            vt = model.predict_v(xt, t, y, m)[:, nT_cond:]
            xt_cat = xt_cat + vt * (ts[i + 1] - ts[i])
        xt = xt_cat
        xt = rearrange(xt, '1 nt nh nw c t h w -> 1 nt c t (nh h) (nw w)')
        
        # VAE decode.
        lq_dec = lq[:, :, nT_cond * tile_min_t:]
        msk_dec = msk_t[:, :, nT_cond * tile_min_t:]
        pred_tail = vae_decode(model.vae, xt, lq=lq_dec, msk=msk_dec)[..., :H, :W]
        pred_tail = rearrange(pred_tail, '1 c t h w -> t c h w').add(1).mul(127.5).clip(0, 255).byte()

        yield output_lq_tail, pred_tail
            
#----------------------------------------------------------------------------

@torch.no_grad()
def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_size = world_size // args.sp_size
    accelerator = Accelerator()
    set_seed(args.seed)
    device = accelerator.device
            
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup distributed training.
    dist_state = str(accelerator.state)
    if args.sp_size > 1:
        sp_state.initialize_sequence_parallel(sequence_parallel_size=args.sp_size)
        sp_group = sp_state.get_sequence_parallel_group()
        sp_world_size = sp_state.get_sequence_parallel_world_size()
        sp_rank = sp_state.get_sequence_parallel_rank()
        sp_src_rank = sp_state.get_sequence_parallel_src_rank()
        dp_rank = sp_state.get_sequence_parallel_group_index()
        dist_state += f"Sequence parallel size: {args.sp_size}\n"
        dist_state += f"Sequence parallel rank: {sp_rank}\n"
    else:
        sp_rank = 0
        dp_rank = accelerator.process_index
    dist_state += f"Data parallel size: {dp_size}\n"
    dist_state += f"Data parallel rank: {dp_rank}\n"
    logger.info(dist_state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error() 

    weight_dtype = torch.bfloat16

    #---------------------------------------------------------------------------------------
    # Initialize the model.

    vae_map = {
        'wan2_1': Wan2_1SpatialTiledEncoder3D,
        'wan2_1_cond': Wan2_1SpatialTiledConditionEncoder3D,
        'wan2_1_cond_v2': Wan2_1SpatialTiledConditionEncoder3Dv2
    }
    tiled_kwargs = dict(
        tile_sample_min_height=args.tile_min_h,
        tile_sample_min_width=args.tile_min_w,
        tile_sample_min_time=args.tile_min_t,
        tile_sample_stride_height=args.tile_stride_h,
        tile_sample_stride_width=args.tile_stride_w,
        spatial_compression_ratio=args.spatial_compression_ratio,
        temporal_compression_ratio=args.temporal_compression_ratio,
    )
    vae = vae_map[args.vae_type](args.vae_path, args.vae_batch_size, **tiled_kwargs)
    vae.init(device)

    if args.model_path is not None:
        logger.info(f"Loading pretrained model from {args.model_path}")
        model_cls = WanTransformer3DModel
        transformer = model_cls.from_pretrained(args.model_path, subfolder="transformer")

        if args.attention_type is not None:
            transformer.set_attention_type(args.attention_type)
            logger.info(f"Set attention type to {args.attention_type}")
                
        lq_encoder = vae_map[args.vae_type](args.vae_path, args.vae_batch_size, **tiled_kwargs)
        lq_encoder.init(device)
        msk_encoder = MaskSpatialTiledEncoder3D(**tiled_kwargs)
        model = Precond(
            transformer=transformer,
            vae=vae,
            lq_encoder=lq_encoder,
            msk_encoder=msk_encoder,
        )

        if args.sp_size > 1:
            model.set_sp_state(sp_group, sp_world_size, sp_rank)

        if accelerator.is_main_process:
            print_module_summary(model, 2)

        model.to(device=device, dtype=weight_dtype).eval()
    
    #------------------------------------------------------------------------------------------------------
    # Main generation loop.

    logger.info("***** Running generation *****")
    logger.info(f"  Sampling steps = {args.sampling_steps}")
    logger.info(f"  Spatial upsampling factor = {args.spatial_sf}")
    logger.info(f"  Temporal upsampling factor = {args.temporal_sf}")
    logger.info(f"  Upsample LQ = {args.upsample_lq}")

    if args.is_img_folder:
        lq_path = [Path(args.lq)]
    else:
        if Path(args.lq).is_file():
            lq_path = [Path(args.lq)]
        else:
            lq_path = [x for x in Path(args.lq).glob("*") if x.suffix.lower() in ['.mp4', '.avi', '.mov']]
            lq_path.sort()

    for i in tqdm(
        range(dp_rank, len(lq_path), dp_size), desc="Total progress", disable=(sp_rank != 0)
    ):
        if args.is_img_folder:
            lq_reader = VideoFrameLazyReader(lq_path[i], dimension_order='NCHW')
        else:
            lq_reader = VideoDecoder(lq_path[i], dimension_order='NCHW')

        gt_reader = None
        if args.lq_downsample > 1:
            gt_reader = lq_reader[:]
            lq_reader = lq_reader[::args.lq_downsample]

        # Create sampler.
        sampler_map = {
            'skip-concat': sample_skip_concat,
            'causal': sample_causal,
        }
        sample_fn = sampler_map[args.sampler](
            lq_reader,
            model,
            train_num_frames=args.num_frames,
            tile_min_t=args.tile_min_t,
            temporal_upsample=args.temporal_upsample,
            spatial_sf=args.spatial_sf,
            temporal_sf=args.temporal_sf,
            nT_cond=1,
            num_steps=args.sampling_steps,
            t_cond=args.t_cond,
            t_shift=args.t_shift,
            device=device,
            weight_dtype=weight_dtype,
            verbose=(sp_rank == 0),
        )

        # Generate.
        lq_list = []
        pred_list = []
        for j, (lq_uint8, pred_uint8) in enumerate(sample_fn):
            if sp_rank == 0:
                lq_list.append(lq_uint8)
                pred_list.append(pred_uint8)
            if args.max_chunks is not None and j + 1 >= args.max_chunks:
                logger.info(f"Reached max chunks: {args.max_chunks}. Stopping generation.")
                break
            
        # Save results.
        if sp_rank == 0:
            output_dir = Path(args.output_dir) / lq_path[i].stem
            os.makedirs(output_dir, exist_ok=True)
            lq_uint8 = rearrange(torch.cat(lq_list, dim=0), 't c h w -> t h w c')
            pred_uint8 = rearrange(torch.cat(pred_list, dim=0), 't c h w -> t h w c')
            codec_kwargs = {
                'video_codec': "libx264",
                'options': {'crf': '10', 'pix_fmt': 'yuv420p'}
            }
            if gt_reader is not None:
                gt_reader = rearrange(torch.cat(gt_reader, dim=0), 't c h w -> t h w c')
                pred_uint8 = pred_uint8[:len(gt_reader)]
                torchvision.io.write_video(os.path.join(output_dir, 'gt.mp4'), pred_uint8, fps=args.fps, **codec_kwargs)
            torchvision.io.write_video(os.path.join(output_dir, 'pred.mp4'), pred_uint8, fps=args.fps, **codec_kwargs)
            if args.save_lq:
                torchvision.io.write_video(os.path.join(output_dir, 'lq.mp4'), lq_uint8, fps=args.fps / args.temporal_sf, **codec_kwargs)
            logger.info(f"Results saved to {output_dir}")
            
    accelerator.end_training()
    
#----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--data",                      type=str,              default=None,            help="Path to the dataset on disk.")
    parser.add_argument("--fps",                       type=float,            default=30,              help="Frames per second of the video.")
    parser.add_argument("--is_img_folder",             action="store_true",   default=False,           help=None)

    # Model arguments
    parser.add_argument("--model_path",                type=str,              default=None,            help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--lq_proj_hidden_dim",        type=int,              default=64,              help="Hidden dim for LQProj (kept for compatibility; weights loaded from EMA).")
    parser.add_argument("--vae_channels",              type=int,              default=None,            help="Number of channels in the VAE.")
    parser.add_argument("--attention_type",            type=str,              default=None,            help="Attention type for the model.")
    parser.add_argument("--temporal_upsample",         type=str,              default='nearest',       help="Temporal upsampling mode.")
    parser.add_argument("--num_frames",                type=int,              default=60,              help="Number of frames of the video.")
    
    # VAE arguments
    parser.add_argument("--vae_type",                  type=str,             default='wan2_1',        choices=['wan2_1', 'wan2_1_cond', 'wan2_1_cond_v2'])
    parser.add_argument("--vae_path",                  type=str,             default=None,            help="Path to VAE.")
    parser.add_argument("--vae_batch_size",            type=int,             default=2,               help="Per-device batch size used inside VAE encode/decode.")
    parser.add_argument("--tile_min_h",                type=int,             default=128,             help="Tile minimum height (pixels).")
    parser.add_argument("--tile_min_w",                type=int,             default=128,             help="Tile minimum width (pixels).")
    parser.add_argument("--tile_min_t",                type=int,             default=20,              help="Tile minimum temporal length (frames). For cosmos, must be divisible by 5.")
    parser.add_argument("--tile_stride_h",             type=int,             default=96,              help="Tile stride height (pixels).")
    parser.add_argument("--tile_stride_w",             type=int,             default=96,              help="Tile stride width (pixels).")
    parser.add_argument("--spatial_compression_ratio", type=int,             default=8,               help="Spatial compression ratio of the VAE.")
    parser.add_argument("--temporal_compression_ratio",type=int,             default=4,               help="Temporal compression ratio of the VAE.")
        
    # Generation arguments
    parser.add_argument("--lq_downsample",             type=int,              default=1,               help=None)
    parser.add_argument("--sampler",                   type=str,              default="skip-concat",   help=None)
    parser.add_argument("--seed",                      type=int,              default=42,              help="Random seed.")
    parser.add_argument("--batch_vae",                 type=int,              default=16,              help="[Deprecated] VAE Batch size per device. Use --vae_batch_size instead.")
    parser.add_argument("--sampling_steps",            type=int,              default=100,             help="Number of denoising steps.")
    parser.add_argument("--t_cond",                    type=float,            default=0.1,             help="t_cond for AR generation.")
    parser.add_argument("--t_shift",                   type=float,            default=1.0,             help="Timestep shift for high resolution.")
    parser.add_argument("--upsample_lq",               type=str,              default='trilinear',     choices=['nearest', 'trilinear', 'pad'], help="Temporal upsample mode for LQ: interpolate or zero-pad frames.")
    parser.add_argument("--spatial_sf",                type=int,              default=1,               help="Spatial upsampling factor.")
    parser.add_argument("--temporal_sf",               type=int,              default=4,               help="Temporal upsampling factor.")
    parser.add_argument("--lq",                        type=str,              default=None,            help="Path to low-quality video for inference.")
    parser.add_argument("--output_dir",                type=str,              default=None,            help="Path to output directory for inference results.")
    parser.add_argument("--max_chunks",                type=int,              default=None,            help="Maximum number of chunks to process during inference.")
    parser.add_argument("--sp_size",                   type=int,              default=1,               help="Sequence parallel size.")
    parser.add_argument("--save_lq",                   action="store_true",   default=False,           help="Save the low-quality and upsampled videos.")
    
    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)