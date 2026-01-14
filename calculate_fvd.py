"""Compute FID and/or FVD for interpolation results.

This standalone CLI scans flattened video folders under --gt-dir and --pred-dir,
matches frames by filename, and computes:
- FID over individual frames (optionally skipping key frames via --skip-step)
- FVD over non-overlapping sequences of length --sequence-length

Notes:
- This script depends on heavy models (Inception for FID, I3D for FVD). Only
  import/run it when you actually need FID/FVD.
- Directory mapping assumes both GT and Pred roots contain scene folders as
  immediate children with matching names, each holding frames like *.png.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.distributed as dist

from metrics.fid_metrics.fid import (
    build_inception,
    calculate_fid,
    postprocess_i2d_pred,
)
from metrics.fvd.fvd import (
    load_fvd_model,
    frechet_distance as i3d_frechet_distance,
    FVD_SAMPLE_SIZE as I3D_FVD_SAMPLE_SIZE,
    MAX_BATCH as I3D_MAX_BATCH,
    TARGET_RESOLUTION as I3D_TARGET_RESOLUTION,
    preprocess as i3d_preprocess,
)


# ---------- Directory helpers ----------

def list_flattened_scene_dirs(root: str) -> List[str]:
    dirs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        try:
            files = os.listdir(p)
        except Exception:
            continue
        if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
            dirs.append(p)
    return dirs


def _basename_dir(p: str) -> str:
    return os.path.basename(os.path.normpath(p))


def map_pred_video_dirs_flat(gt_root: str, pred_root: str) -> Dict[str, str]:
    gt_dirs = list_flattened_scene_dirs(gt_root)
    pred_dirs = list_flattened_scene_dirs(pred_root)
    gt_index = {_basename_dir(p): p for p in gt_dirs}
    pred_index = {_basename_dir(p): p for p in pred_dirs}
    names = sorted(set(gt_index.keys()) & set(pred_index.keys()))
    return {gt_index[n]: pred_index[n] for n in names}


def collect_frames(dir_path: str, ext: str) -> List[str]:
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.' + ext.lower())])
    return [os.path.join(dir_path, f) for f in files]


def _parse_frame_index(filename: str) -> int | None:
    stem = os.path.splitext(filename)[0]
    m = re.findall(r"(\d+)", stem)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def match_frames(pred_dir: str, gt_dir: str, ext: str, skip_step: int = 0) -> List[Tuple[str, str]]:
    pred_frames = collect_frames(pred_dir, ext)
    gt_frames = collect_frames(gt_dir, ext)
    pred_map = {Path(p).name: p for p in pred_frames}
    pairs = []
    for g in gt_frames:
        name = Path(g).name
        if name in pred_map:
            if skip_step and skip_step > 0:
                idx = _parse_frame_index(name)
                if idx is not None and idx % skip_step == 0:
                    continue
            pairs.append((pred_map[name], g))
    return pairs


# ---------- Image/sequence I/O ----------

def _load_image(path: str, transform, device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if transform:
        img = transform(img)
    return img.to(device)


def _sample_non_overlap_sequences(paths: List[str], seq_len: int) -> List[List[str]]:
    return [paths[i:i+seq_len] for i in range(0, len(paths) - seq_len + 1, seq_len)]


def _make_numpy_sequence(seq: List[str]) -> np.ndarray:
    frames = []
    for p in seq:
        img = Image.open(p).convert('RGB')
        frames.append(np.array(img, dtype=np.uint8))  # H,W,C
    return np.stack(frames, axis=0)  # T,H,W,C


def _stack_i3d_logits(video_tensors: List[torch.Tensor], i3d, device: torch.device) -> torch.Tensor:
    """Compute I3D logits in batches from a list of (1,C,T,H,W) tensors."""
    if not video_tensors:
        return torch.empty(0, 400)
    # Concatenate up to MAX_BATCH videos per forward
    logits_parts = []
    for i in range(0, len(video_tensors), I3D_MAX_BATCH):
        batch = torch.cat(video_tensors[i:i+I3D_MAX_BATCH], dim=0).to(device)  # (B,C,T,H,W)
        with torch.no_grad():
            logits_parts.append(i3d(batch).cpu())
    return torch.cat(logits_parts, dim=0)


# ---------- Metrics ----------

def compute_fid_features(pred_video_map: Dict[str, str], transform, accelerator: Accelerator, dims: int, batch_size: int, ext: str, skip_step: int):
    device = accelerator.device
    model = build_inception(dims).to(device).eval()
    feats_pred = []
    feats_gt = []

    # Precompute frame pairs per video to know progress total and avoid recomputing
    pairs_per_video = {}
    total_items = 0
    for gt_dir, pred_dir in pred_video_map.items():
        frame_pairs = match_frames(pred_dir, gt_dir, ext, skip_step=skip_step)
        pairs_per_video[(gt_dir, pred_dir)] = frame_pairs
        total_items += len(frame_pairs) * 2  # pred + gt passes

    pbar = tqdm(total=total_items, desc='FID', dynamic_ncols=True) if accelerator.is_local_main_process else None

    with torch.no_grad():
        for (gt_dir, pred_dir), frame_pairs in pairs_per_video.items():
            for which in ['pred', 'gt']:
                paths = [p if which == 'pred' else g for p, g in frame_pairs]
                for i in range(0, len(paths), batch_size):
                    batch_paths = paths[i:i+batch_size]
                    imgs = [_load_image(bp, transform, device) for bp in batch_paths]
                    if not imgs:
                        continue
                    x = torch.stack(imgs, 0)
                    pred = model(x)
                    pred = postprocess_i2d_pred(pred)
                    # Ensure 2D shape (B, D) even for single-image batches
                    if pred.ndim == 1:
                        pred = pred.unsqueeze(0)
                    elif pred.ndim > 2:
                        pred = pred.flatten(1)
                    if which == 'pred':
                        feats_pred.append(pred.cpu())
                    else:
                        feats_gt.append(pred.cpu())
                    if pbar is not None:
                        pbar.update(len(batch_paths))
    if pbar is not None:
        pbar.close()
    if not feats_pred or not feats_gt:
        # Return empty tensors to allow safe gather
        return torch.empty((0, dims), dtype=torch.float32), torch.empty((0, dims), dtype=torch.float32)
    act1 = torch.cat(feats_pred, 0).cpu()
    act2 = torch.cat(feats_gt, 0).cpu()
    return act1, act2


def compute_fid_all(pred_video_map: Dict[str, str], transform, accelerator: Accelerator, dims: int, batch_size: int, ext: str, skip_step: int) -> float:
    # Single-process convenience
    act1, act2 = compute_fid_features(pred_video_map, transform, accelerator, dims, batch_size, ext, skip_step)
    if act1.numel() == 0 or act2.numel() == 0:
        return float('nan')
    return float(calculate_fid(act1.numpy(), act2.numpy()))


def compute_fvd_logits(pred_video_map: Dict[str, str], accelerator: Accelerator, seq_len: int, ext: str):
    if load_fvd_model is None or i3d_frechet_distance is None:
        raise RuntimeError('I3D FVD components unavailable.')
    device = accelerator.device
    i3d = load_fvd_model(device)
    real_embs = []
    fake_embs = []
    # Buffers accumulate per-rank sequences prior to model forward
    fake_buf: List[torch.Tensor] = []  # each (1,C,T,H,W)
    real_buf: List[torch.Tensor] = []
    # Pre-scan to determine total number of valid sequences for progress only (no flooring)
    total_available = 0
    for gt_dir, pred_dir in pred_video_map.items():
        frame_pairs = match_frames(pred_dir, gt_dir, ext, skip_step=0)
        preds = [p for p, _ in frame_pairs]
        gts = [g for _, g in frame_pairs]
        n_pred = max(0, (len(preds) - seq_len + 1) // seq_len)
        n_gt = max(0, (len(gts) - seq_len + 1) // seq_len)
        total_available += min(n_pred, n_gt)
    planned_total = total_available  # show full local count; truncation will happen on rank0 later
    pbar = tqdm(total=planned_total, desc='FVD', dynamic_ncols=True) if (planned_total > 0 and accelerator.is_local_main_process) else None
    processed = 0
    for gt_dir, pred_dir in pred_video_map.items():
        frame_pairs = match_frames(pred_dir, gt_dir, ext, skip_step=0)  # ignore skip for FVD
        preds = [p for p, _ in frame_pairs]
        gts = [g for _, g in frame_pairs]
        pred_seqs = _sample_non_overlap_sequences(preds, seq_len)
        gt_seqs = _sample_non_overlap_sequences(gts, seq_len)
        n_seqs = min(len(pred_seqs), len(gt_seqs))
        pred_seqs = pred_seqs[:n_seqs]
        gt_seqs = gt_seqs[:n_seqs]
        for ps, gs in zip(pred_seqs, gt_seqs):
            fake_np = _make_numpy_sequence(ps)  # T,H,W,C uint8
            real_np = _make_numpy_sequence(gs)
            # Per-sequence preprocess -> (1,C,T,H,W) float tensor
            fake_tensor = i3d_preprocess(fake_np[None, ...], I3D_TARGET_RESOLUTION)
            real_tensor = i3d_preprocess(real_np[None, ...], I3D_TARGET_RESOLUTION)
            fake_buf.append(fake_tensor)
            real_buf.append(real_tensor)
            if pbar is not None:
                processed += 1
                pbar.update(1)
            # When we have a full batch, run the network and reset buffers
            if len(fake_buf) == I3D_MAX_BATCH:
                fake_embs.append(_stack_i3d_logits(fake_buf, i3d, device))
                real_embs.append(_stack_i3d_logits(real_buf, i3d, device))
                fake_buf.clear(); real_buf.clear()
    # Process any leftover sequences that didn't fill a full MAX_BATCH
    if fake_buf or real_buf:
        fake_embs.append(_stack_i3d_logits(fake_buf, i3d, device))
        real_embs.append(_stack_i3d_logits(real_buf, i3d, device))
        fake_buf.clear(); real_buf.clear()
    if pbar is not None:
        pbar.close()
    if not fake_embs or not real_embs:
        print('[WARN] No sequences for FVD on this rank; returning empty logits')
        return torch.empty(0, 400), torch.empty(0, 400)
    fake_logits = torch.cat(fake_embs, 0).cpu()
    real_logits = torch.cat(real_embs, 0).cpu()
    # Subsample will be applied later on rank0 after gathering
    return fake_logits, real_logits


def compute_fvd_all(pred_video_map: Dict[str, str], accelerator: Accelerator, seq_len: int, ext: str) -> float:
    # Single-process convenience
    fake_logits, real_logits = compute_fvd_logits(pred_video_map, accelerator, seq_len, ext)
    if fake_logits.numel() == 0 or real_logits.numel() == 0:
        return float('nan')
    sample_size = min(I3D_FVD_SAMPLE_SIZE, fake_logits.shape[0], real_logits.shape[0])
    fake_logits = fake_logits[:sample_size]
    real_logits = real_logits[:sample_size]
    return float(i3d_frechet_distance(fake_logits, real_logits).item())


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description='Compute FID and/or FVD for VFI outputs')
    p.add_argument('--pred-dir', type=str, required=True, help='Prediction root directory')
    p.add_argument('--gt-dir', type=str, required=True, help='Ground truth root directory')
    p.add_argument('--image-ext', type=str, default='png', help='Image extension to scan')
    p.add_argument('--skip-step', type=int, default=0, help='Skip key frames by idx % skip_step == 0 when computing FID')
    p.add_argument('--fid-dims', type=int, default=2048, help='Inception feature dims for FID')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--sequence-length', type=int, default=16, help='Sequence length for FVD (I3D expects 16)')
    p.add_argument('--max-videos', type=int, default=None, help='Optional cap for debugging')
    p.add_argument('--fid', action='store_true', help='Compute FID')
    p.add_argument('--fvd', action='store_true', help='Compute FVD')
    p.add_argument('--output-json', type=str, default=None, help='Write results to JSON')
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        print('Args:', args)
        try:
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(accelerator.device) if cuda_ok else 'CPU'
        except Exception:
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU'
        print(f'Device: {accelerator.device} | CUDA available: {cuda_ok} | GPU: {gpu_name}')

    # If neither flag set, compute both by default
    compute_fid_flag = args.fid or (not args.fvd)
    compute_fvd_flag = args.fvd or (not args.fid)

    with accelerator.main_process_first():
        mapping = map_pred_video_dirs_flat(args.gt_dir, args.pred_dir)
        gt_video_dirs = sorted(mapping.keys())
        if args.max_videos:
            gt_video_dirs = gt_video_dirs[:args.max_videos]
            mapping = {k: mapping[k] for k in gt_video_dirs}
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    # Shard videos across processes for data-parallel computation
    gt_video_dirs_local = gt_video_dirs[rank::world_size] if world_size > 1 else gt_video_dirs
    mapping_local = {k: mapping[k] for k in gt_video_dirs_local}
    if accelerator.is_local_main_process:
        print(f'Found {len(gt_video_dirs)} GT videos, {len(mapping)} with predictions. world_size={world_size}')

    transform = T.ToTensor()

    results = {}
    if compute_fid_flag:
        # Compute local features then gather across processes
        act1_local, act2_local = compute_fid_features(mapping_local, transform, accelerator, args.fid_dims, args.batch_size, args.image_ext, args.skip_step)
        # Convert to numpy for object gathering (variable lengths)
        act1_np = act1_local.numpy()
        act2_np = act2_local.numpy()
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gather_pred = [None for _ in range(world_size)]
            gather_gt = [None for _ in range(world_size)]
            dist.all_gather_object(gather_pred, act1_np)
            dist.all_gather_object(gather_gt, act2_np)
            if accelerator.is_local_main_process:
                act1_all = np.concatenate([a for a in gather_pred if a is not None and a.size > 0], axis=0) if any((a is not None and a.size > 0) for a in gather_pred) else np.empty((0, act1_np.shape[1] if act1_np.ndim == 2 else 2048))
                act2_all = np.concatenate([a for a in gather_gt if a is not None and a.size > 0], axis=0) if any((a is not None and a.size > 0) for a in gather_gt) else np.empty((0, act2_np.shape[1] if act2_np.ndim == 2 else 2048))
                fid_value = float('nan') if act1_all.shape[0] == 0 or act2_all.shape[0] == 0 else float(calculate_fid(act1_all, act2_all))
            else:
                fid_value = None
            # Broadcast result to all
            fid_tensor = torch.tensor([fid_value if fid_value is not None and fid_value == fid_value else -1.0], device=accelerator.device)
            dist.broadcast(fid_tensor, src=0)
            fid_value = fid_tensor.item()
            fid_value = float('nan') if fid_value < 0 else float(fid_value)
        else:
            # Single process
            fid_value = float('nan') if act1_np.shape[0] == 0 or act2_np.shape[0] == 0 else float(calculate_fid(act1_np, act2_np))
        results['fid'] = fid_value
        if accelerator.is_local_main_process:
            print(f'FID: {fid_value:.4f}' if fid_value == fid_value else 'FID: NaN')
    if compute_fvd_flag:
        fake_logits_local, real_logits_local = compute_fvd_logits(mapping_local, accelerator, args.sequence_length, args.image_ext)
        # Convert to numpy for gathering
        fake_np = fake_logits_local.numpy()
        real_np = real_logits_local.numpy()
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gather_fake = [None for _ in range(world_size)]
            gather_real = [None for _ in range(world_size)]
            dist.all_gather_object(gather_fake, fake_np)
            dist.all_gather_object(gather_real, real_np)
            if accelerator.is_local_main_process:
                fake_all = np.concatenate([a for a in gather_fake if a is not None and a.size > 0], axis=0) if any((a is not None and a.size > 0) for a in gather_fake) else np.empty((0, fake_np.shape[1] if fake_np.ndim == 2 else 400))
                real_all = np.concatenate([a for a in gather_real if a is not None and a.size > 0], axis=0) if any((a is not None and a.size > 0) for a in gather_real) else np.empty((0, real_np.shape[1] if real_np.ndim == 2 else 400))
                sample_size = min(I3D_FVD_SAMPLE_SIZE, fake_all.shape[0], real_all.shape[0])
                if sample_size > 0:
                    fake_all = torch.from_numpy(fake_all[:sample_size])
                    real_all = torch.from_numpy(real_all[:sample_size])
                    fvd_value = float(i3d_frechet_distance(fake_all, real_all).item())
                else:
                    fvd_value = float('nan')
            else:
                fvd_value = None
            fvd_tensor = torch.tensor([fvd_value if fvd_value is not None and fvd_value == fvd_value else -1.0], device=accelerator.device)
            dist.broadcast(fvd_tensor, src=0)
            fvd_value = fvd_tensor.item()
            fvd_value = float('nan') if fvd_value < 0 else float(fvd_value)
        else:
            # Single process
            sample_size = min(I3D_FVD_SAMPLE_SIZE, fake_np.shape[0], real_np.shape[0])
            if sample_size > 0:
                fvd_value = float(i3d_frechet_distance(torch.from_numpy(fake_np[:sample_size]), torch.from_numpy(real_np[:sample_size])).item())
            else:
                fvd_value = float('nan')
        results['fvd'] = fvd_value
        if accelerator.is_local_main_process:
            print(f'FVD: {fvd_value:.4f}' if fvd_value == fvd_value else 'FVD: NaN')

    if accelerator.is_local_main_process and args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote JSON: {args.output_json}')


if __name__ == '__main__':
    main()
