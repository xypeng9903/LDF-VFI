from torch import nn
import torch
from einops import rearrange, repeat
from pathlib import Path
import torch.nn.functional as F
import torch.distributed as dist
import os
from copy import deepcopy
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .transformer_wan import WanTransformer3DModel
from .vae2_1 import WanVAE
from .msk_encoder import MaskEncoder

#-----------------------------------------------------------------

class Precond(nn.Module):
    def __init__(self, 
        transformer: WanTransformer3DModel, 
        vae,
        lq_encoder, 
        msk_encoder,
        diffusion_forcing='temporal',
        t_shift=1.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.lq_encoder = lq_encoder
        self.msk_encoder = msk_encoder
        self.vae = vae
        self.use_sp = False
        
        # EMA state.
        self._ema_decays = [] 
        self._ema_states = [] 
        self._ema_device = 'cpu'

        t_sampler_map = {
            'all': lambda leading_shape: torch.randn(leading_shape).sigmoid_(),
            'temporal': lambda leading_shape: torch.randn((*leading_shape[0:2], 1, 1)).sigmoid_().expand(leading_shape).contiguous(),
        }
        self.t_sampler = t_sampler_map[diffusion_forcing]
        self.t_shift = t_shift
    
    def set_sp_state(self, sp_group, sp_world_size, sp_rank, sp_src_rank):
        def fn(module):
            if hasattr(module, "_set_sp_state"):
                module._set_sp_state(
                    sp_group=sp_group, 
                    sp_world_size=sp_world_size, 
                    sp_rank=sp_rank,
                    sp_src_rank=sp_src_rank,
                )
                
        self.apply(fn)

    def _set_sp_state(self, sp_group, sp_world_size, sp_rank, sp_src_rank):
        self.use_sp = True
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.sp_rank = sp_rank
        self.sp_src_rank = sp_src_rank

    def predict_v(self, xt, t, y, msk):
        t = t.mul(1000).to(dtype=torch.long)
        model_input = torch.cat([xt, y, msk], dim=-4)
        model_output = self.transformer(model_input, t).sample
        return model_output

    def forward(self, y: torch.Tensor, msk: torch.Tensor, x0: torch.Tensor, pre_encode=False):
        dtype, device = x0.dtype, x0.device

        # Encode latents.
        if not pre_encode:
            with torch.no_grad():
                y = self.lq_encoder.encode(y, for_train=True).contiguous()
                msk = self.msk_encoder.encode(msk, for_train=True).contiguous()
                x0 = self.vae.encode(x0, for_train=True).contiguous()
        
        # Sample noise and timesteps.    
        x1 = torch.randn_like(x0)
        leading_shape = x0.shape[:-4]
        t = self.t_sampler(leading_shape).to(dtype=dtype, device=device)
        t = self.t_shift * t / (1 + (self.t_shift - 1) * t)
        if self.use_sp:
            dist.broadcast(x1, src=self.sp_src_rank, group=self.sp_group)
            dist.broadcast(t, src=self.sp_src_rank, group=self.sp_group)
        w = t.view(*leading_shape, 1, 1, 1, 1)
        xt = x0 * (1 - w) + x1 * w
        target = x1 - x0

        # Compute loss.
        model_output = self.predict_v(xt, t, y, msk)
        loss = F.mse_loss(model_output.float(), target.float(), reduction='none')
        return loss

    #---------------------------------------------
    # EMA utilities.

    @torch.no_grad()
    def prepare_ema(self, ema_decays, save_dir=None, device='cpu'):
        self._ema_decays = list(float(d) for d in ema_decays if d is not None)
        self._ema_states = []
        self._ema_device = device
        for decay in self._ema_decays:
            try:
                save_path = os.path.join(save_dir, f"ema-{decay}")
                transformer = self.transformer.__class__.from_pretrained(save_path, subfolder="transformer")
            except:
                transformer = deepcopy(self.transformer)
            self._ema_states.append({
                'transformer': transformer.to(device=device, dtype=torch.float32).requires_grad_(False),
            })

    @torch.no_grad()
    def update_ema(self, state_dict):
        if not self._ema_states:
            return
        for decay, ema_state in zip(self._ema_decays, self._ema_states):
            for module_name, ema_module in ema_state.items():
                # Parameters
                for pname, ema_param in ema_module.named_parameters():
                    key = f"{module_name}.{pname}"
                    src = state_dict[key]
                    src = src.to(device=self._ema_device, dtype=torch.float32)
                    ema_param.data.mul_(decay).add_(src, alpha=1 - decay)
                # Buffers
                for bname, ema_buf in ema_module.named_buffers():
                    key = f"{module_name}.{bname}"
                    src = state_dict[key]
                    ema_buf.data.copy_(src.to(device=self._ema_device, dtype=ema_buf.dtype))

    @torch.no_grad()
    def save_ema(self, save_dir: str):
        for decay, ema_state in zip(self._ema_decays, self._ema_states):
            ema_dir = Path(save_dir) / f"ema-{decay}"
            ema_dir.mkdir(parents=True, exist_ok=True)
            for name, module in ema_state.items():
                module.save_pretrained(ema_dir / name)

#-----------------------------------------------------------------
# Spatial overlapped local VAE. 
# TODO: sequence parrallel

class SpatialTiledEncoder3D:
    def __init__(self,
        tile_sample_min_height=128,
        tile_sample_min_width=128,
        tile_sample_min_time=20,
        tile_sample_stride_height=96,
        tile_sample_stride_width=96,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
    ):
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_min_time = tile_sample_min_time
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., :, x] = a[..., :, -blend_extent + x] * (1 - x / blend_extent) + b[..., :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, x.shape[-2], self.tile_sample_stride_height):
            row = []
            for j in range(0, x.shape[-1], self.tile_sample_stride_width):
                tile = x[..., i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                pad_h = self.tile_sample_min_height - tile.shape[-2]
                pad_w = self.tile_sample_min_width - tile.shape[-1]
                tile = F.pad(tile, (0, pad_w, 0, pad_h))
                tile = self._encode(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[..., :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.stack(result_row))
        encoded = torch.stack(result_rows)
        return encoded
    
    def tiled_decode(self, z: torch.Tensor)-> torch.Tensor:
        height, width = z.shape[-2:]

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                tile = z[..., i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                decoded = self._decode(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[..., : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        decoded = torch.cat(result_rows, dim=-2)
        return decoded
    
    def _encode(self, x):
        raise NotImplementedError(f"class {self.__class__.__name__} should implement `_encode` method.")
    
    def _decode(self, z):
        raise NotImplementedError(f"class {self.__class__.__name__} should implement `_decode` method.")
    
    def encode(self, x, for_train=False):
        assert x.shape[2] % self.tile_sample_min_time == 0, \
            f"Input T={x.shape[2]} must be divisible by tile_sample_min_time={self.tile_sample_min_time}."
        B = x.shape[0]
        height, width = x.shape[-2:]
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        x = rearrange(x, "b c (nt t) h w -> (b nt) c t h w", t=self.tile_sample_min_time)
        x = self.tiled_encode(x)

        if for_train:
            x = rearrange(x, "nh nw (b nt) c t h w -> b nt nh nw c t h w", b=B)
        else:
            x = rearrange(x, "nh nw (b nt) c t h w -> b nt c t (nh h) (nw w)", b=B)
            x = x[..., : latent_height, : latent_width]
        return x

    def decode(self, z):
        B = z.shape[0]
        z = rearrange(z, "b nt c t h w -> (b nt) c t h w")
        z = self.tiled_decode(z)
        z = rearrange(z, "(b nt) c t h w -> b c (nt t) h w", b=B)
        return z


class Wan2_1SpatialTiledEncoder3D(SpatialTiledEncoder3D):
    def __init__(self, vae_path, batch_size, **kwargs):
        assert kwargs.get('tile_sample_min_time') % 5 == 0, "tile_sample_min_time must be divisible by 5 for CosmosSpatialTiledEncoder3D."
        super().__init__(**kwargs)
        self.vae_path = vae_path
        self.vae = None
        self.batch_size = batch_size
        self.dtype = torch.bfloat16
    
    def init(self, device):
        if self.vae is None:
            self.vae = WanVAE(vae_pth=self.vae_path, device=device, dtype=self.dtype)
            self.vae.model.to(device=device, dtype=self.dtype)
    
    def _encode(self, x):
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        scale = self.vae.scale
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        x = torch.cat([self.vae.model.encode(batch, scale) for batch in x.split(self.batch_size)])
        x = rearrange(x, "(k b) c t h w -> b c (k t) h w", b=bsz).to(dtype)
        return x

    def _decode(self, x):        
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        scale = self.vae.scale
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=2).to(self.dtype)
        x = torch.cat([self.vae.model.decode(batch, scale) for batch in x.split(self.batch_size)])
        x = rearrange(x, "(k b) c t h w -> b c (k t) h w", b=bsz).clamp(-1, 1).to(dtype)
        return x
        

from .utils import LQProj

class LQProjSpatialTiledEncoder3D(ModelMixin, ConfigMixin, SpatialTiledEncoder3D):
    @register_to_config
    def __init__(
        self,
        hidden_dim: int = 64,
        batch_size: int = 2,
        tile_sample_min_height: int = 128,
        tile_sample_min_width: int = 128,
        tile_sample_min_time: int = 20,
        tile_sample_stride_height: int = 96,
        tile_sample_stride_width: int = 96,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
    ):
        nn.Module.__init__(self)
        SpatialTiledEncoder3D.__init__(
            self,
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_min_time=tile_sample_min_time,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
            spatial_compression_ratio=spatial_compression_ratio,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        # We chunk temporal windows with t=5 in _encode, so enforce divisibility like Cosmos encoder
        assert tile_sample_min_time % 5 == 0, (
            "tile_sample_min_time must be divisible by 5 for LQProjSpatialTiledEncoder3D when using 5-frame chunking."
        )
        self.batch_size = batch_size
        self.proj = LQProj(
            ff=1,
            hh=spatial_compression_ratio,
            ww=spatial_compression_ratio,
            in_dim=3,
            hidden_dim1=hidden_dim,
            hidden_dim2=hidden_dim,
        )

    def _encode(self, x):
        bsz = x.shape[0]
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=5)
        x = torch.cat([self.proj(batch) for batch in x.split(self.batch_size)])
        x = rearrange(x, "(k b) c t h w -> b c (k t) h w", b=bsz)
        return x
        
    def _decode(self, z):
        raise NotImplementedError("LQProjSpatialTiledEncoder3D does not support decode().")

    

class MaskSpatialTiledEncoder3D(SpatialTiledEncoder3D):
    def __init__(self, **kwargs):
        assert kwargs.get('tile_sample_min_time') % 5 == 0, "tile_sample_min_time must be divisible by 5 for MaskSpatialTiledEncoder3D."
        super().__init__(**kwargs)
        self.mask_encoder = MaskEncoder(kwargs.get('spatial_compression_ratio'))
    
    def _encode(self, x):
        bsz = x.shape[0]
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=5)
        x = self.mask_encoder(x)
        x = rearrange(x, "(k b) c t h w -> b c (k t) h w", b=bsz)
        return x

    def _decode(self, z):
        raise NotImplementedError("MaskSpatialTiledEncoder3D does not implement decode.")

#-----------------------------------------------------------------
# Spatial overlapped local conditional VAE. 
# TODO: sequence parrallel

class SpatialTiledConditionEncoder3D:
    def __init__(self,
        tile_sample_min_height=128,
        tile_sample_min_width=128,
        tile_sample_min_time=20,
        tile_sample_stride_height=96,
        tile_sample_stride_width=96,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
    ):
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_sample_min_time = tile_sample_min_time
        self.tile_sample_stride_height = tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width
        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., :, x] = a[..., :, -blend_extent + x] * (1 - x / blend_extent) + b[..., :, x] * (x / blend_extent)
        return b
    
    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, x.shape[-2], self.tile_sample_stride_height):
            row = []
            for j in range(0, x.shape[-1], self.tile_sample_stride_width):
                tile = x[..., i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                pad_h = self.tile_sample_min_height - tile.shape[-2]
                pad_w = self.tile_sample_min_width - tile.shape[-1]
                tile = F.pad(tile, (0, pad_w, 0, pad_h))
                tile = self._encode(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[..., :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.stack(result_row))
        encoded = torch.stack(result_rows)
        return encoded
    
    def tiled_decode(self, z: torch.Tensor, cond, msk)-> torch.Tensor:
        height, width = z.shape[-2:]

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                z_tile = z[..., i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                cond_tile = cond[
                    ..., 
                    i * self.spatial_compression_ratio : (i + tile_latent_min_height) * self.spatial_compression_ratio, 
                    j * self.spatial_compression_ratio : (j + tile_latent_min_width) * self.spatial_compression_ratio
                ]
                msk_tile = msk[
                    ..., 
                    i * self.spatial_compression_ratio : (i + tile_latent_min_height) * self.spatial_compression_ratio, 
                    j * self.spatial_compression_ratio : (j + tile_latent_min_width) * self.spatial_compression_ratio
                ]
                decoded = self._decode(z_tile, cond_tile, msk_tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[..., : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        decoded = torch.cat(result_rows, dim=-2)
        return decoded
        
    def _encode(self, x):
        raise NotImplementedError(f"class {self.__class__.__name__} should implement `_encode` method.")
    
    def _decode(self, z):
        raise NotImplementedError(f"class {self.__class__.__name__} should implement `_decode` method.")
    
    def encode(self, x, for_train=False):
        assert x.shape[2] % self.tile_sample_min_time == 0, \
            f"Input T={x.shape[2]} must be divisible by tile_sample_min_time={self.tile_sample_min_time}."
        B = x.shape[0]
        height, width = x.shape[-2:]
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        x = rearrange(x, "b c (nt t) h w -> (b nt) c t h w", t=self.tile_sample_min_time)
        x = self.tiled_encode(x)

        if for_train:
            x = rearrange(x, "nh nw (b nt) c t h w -> b nt nh nw c t h w", b=B)
        else:
            x = rearrange(x, "nh nw (b nt) c t h w -> b nt c t (nh h) (nw w)", b=B)
            x = x[..., : latent_height, : latent_width]
        return x

    def decode(self, z, cond, msk):
        B = z.shape[0]
        z = rearrange(z, "b nt c t h w -> (b nt) c t h w")
        cond = rearrange(cond, "b nt c t h w -> (b nt) c t h w")
        msk = rearrange(msk, "b nt c t h w -> (b nt) c t h w")
        z = self.tiled_decode(z, cond, msk)
        z = rearrange(z, "(b nt) c t h w -> b c (nt t) h w", b=B)
        return z
    

from .vae2_1_cond import WanVAECondition_
    
class Wan2_1SpatialTiledConditionEncoder3D(SpatialTiledConditionEncoder3D):
    def __init__(self, vae_path, batch_size, **kwargs):
        assert kwargs.get('tile_sample_min_time') % 5 == 0, "tile_sample_min_time must be divisible by 5 for CosmosSpatialTiledEncoder3D."
        super().__init__(**kwargs)
        self.vae_path = vae_path
        self.vae = None
        self.batch_size = batch_size
        self.dtype = torch.bfloat16
    
    def init(self, device):
        if self.vae is None:
            cfg = dict(
                dim=96,
                z_dim=16,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0
            )
            self.vae = WanVAECondition_(**cfg)
            print(self.vae.load_state_dict(torch.load(self.vae_path, map_location=device), strict=False))
            self.vae.to(device=device, dtype=self.dtype).requires_grad_(False).eval()

            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
            self.mean = torch.tensor(mean, dtype=self.dtype, device=device)
            self.std = torch.tensor(std, dtype=self.dtype, device=device)
            self.scale = [self.mean, 1.0 / self.std]

    def _encode(self, x):
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        out = []
        for _x in x.split(self.batch_size):
            mu, logvar = self.vae.encode(_x, self.scale, return_logvar=True)
            z_ = self.vae.reparameterize(mu, logvar)
            z_ = (z_ - self.scale[0].view(1, -1, 1, 1, 1)) * self.scale[1].view(1, -1, 1, 1, 1)
            out.append(z_)
        out = torch.cat(out)
        out = rearrange(out, "(k b) c t h w -> b c (k t) h w", b=bsz).to(dtype)
        return out
    
    def _decode(self, x, cond, msk):        
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=2).to(self.dtype)
        cond = rearrange(cond, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        msk = rearrange(msk, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        out = []
        for _z, _c, _m in zip(x.split(self.batch_size), cond.split(self.batch_size), msk.split(self.batch_size)):
            add_feats = self.vae.encode_cond(_c, _m)
            x_recon = self.vae.decode(_z, add_feats, scale=self.scale)
            out.append(x_recon)
        out = torch.cat(out)
        out = rearrange(out, "(k b) c t h w -> b c (k t) h w", b=bsz).clamp(-1, 1).to(dtype)
        return out

from .vae2_1_cond_v2 import WanVAECondition_ as WanVAECondition_v2
    
class Wan2_1SpatialTiledConditionEncoder3Dv2(SpatialTiledConditionEncoder3D):
    def __init__(self, vae_path, batch_size, **kwargs):
        assert kwargs.get('tile_sample_min_time') % 5 == 0, "tile_sample_min_time must be divisible by 5 for CosmosSpatialTiledEncoder3D."
        super().__init__(**kwargs)
        self.vae_path = vae_path
        self.vae = None
        self.batch_size = batch_size
        self.dtype = torch.bfloat16
    
    def init(self, device):
        if self.vae is None:
            cfg = dict(
                dim=96,
                z_dim=16,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0
            )
            self.vae = WanVAECondition_v2(**cfg)
            print(self.vae.load_state_dict(torch.load(self.vae_path, map_location=device), strict=False))
            self.vae.to(device=device, dtype=self.dtype).requires_grad_(False).eval()

            mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]
            std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]
            self.mean = torch.tensor(mean, dtype=self.dtype, device=device)
            self.std = torch.tensor(std, dtype=self.dtype, device=device)
            self.scale = [self.mean, 1.0 / self.std]

    def _encode(self, x):
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        out = []
        for _x in x.split(self.batch_size):
            mu, logvar = self.vae.encode(_x, self.scale, return_logvar=True)
            z_ = self.vae.reparameterize(mu, logvar)
            z_ = (z_ - self.scale[0].view(1, -1, 1, 1, 1)) * self.scale[1].view(1, -1, 1, 1, 1)
            out.append(z_)
        out = torch.cat(out)
        out = rearrange(out, "(k b) c t h w -> b c (k t) h w", b=bsz).to(dtype)
        return out
    
    def _decode(self, x, cond, msk):        
        self.init(x.device)
        bsz = x.shape[0]
        dtype = x.dtype
        x = rearrange(x, "b c (k t) h w -> (k b) c t h w", t=2).to(self.dtype)
        cond = rearrange(cond, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        msk = rearrange(msk, "b c (k t) h w -> (k b) c t h w", t=5).to(self.dtype)
        out = []
        for _z, _c, _m in zip(x.split(self.batch_size), cond.split(self.batch_size), msk.split(self.batch_size)):
            add_feats = self.vae.encode_cond(_c, _m)
            x_recon = self.vae.decode(_z, add_feats, scale=self.scale)
            out.append(x_recon)
        out = torch.cat(out)
        out = rearrange(out, "(k b) c t h w -> b c (k t) h w", b=bsz).clamp(-1, 1).to(dtype)
        return out