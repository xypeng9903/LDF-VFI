# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import lru_cache, partial
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from training.distributed.util import all_to_all, gather_forward
import os


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

torch._dynamo.config.cache_size_limit = 1000
flex_attention = torch.compile(flex_attention, dynamic=False)

# https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459
if os.environ.get("GPU_TYPE") in ['4090', '3090']:
    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    }
    flex_attention = partial(flex_attention, kernel_options=kernel_options)


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.use_sp = False

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[BlockMask] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D, C = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b n d c -> (b n) d c')
        dtype = hidden_states.dtype
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        dtype = hidden_states.dtype
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).to(dtype)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).to(dtype)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).to(dtype)

        if rotary_emb is not None:
            # qkv: (b n) h d c
            # rotary_emb: n d c/2 
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_rotated = rearrange(x_rotated, '(b n) h d c -> b n h d c', b=B)
                freqs = rearrange(freqs, 'n d c -> 1 n 1 d c')
                x_out = torch.view_as_real(x_rotated * freqs).flatten(-2, -1)
                x_out = x_out.type_as(hidden_states)
                x_out = rearrange(x_out, 'b n h d c -> (b n) h d c')
                return x_out

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # TODO: I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            # key_img = attn.add_k_proj(encoder_hidden_states_img)
            # key_img = attn.norm_added_k(key_img)
            # value_img = attn.add_v_proj(encoder_hidden_states_img)

            # key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            # value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # hidden_states_img = flex_attention(query, key_img, value_img)
            # hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            # hidden_states_img = hidden_states_img.type_as(query)
            raise NotImplementedError
        
        def _attn_preprocess(*args):
            out = []
            for x in args:
                if self.use_sp:
                    x = all_to_all(x, 1, 2, self.sp_group)
                x = rearrange(x, '(b n) h d c -> b h (n d) c', n=N)
                out.append(x)
            return tuple(out)
                
        def _attn_postprocess(x):
            x = rearrange(x, 'b h (n d) c -> (b n) h d c', n=N)
            if self.use_sp:
                x = all_to_all(x, 2, 1, self.sp_group)
            return x
            
        query, key, value = _attn_preprocess(query, key, value)
        hidden_states = flex_attention(query, key, value, block_mask=attention_mask)   
        hidden_states = _attn_postprocess(hidden_states)
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            # hidden_states = hidden_states + hidden_states_img
            raise NotImplementedError

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = rearrange(hidden_states, '(b n) d c -> b n d c', b=B)
        return hidden_states

    def _set_sp_state(self, sp_group, sp_world_size, sp_rank, sp_src_rank):
        self.use_sp = True
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.sp_rank = sp_rank
        self.sp_src_rank = sp_src_rank


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: Optional[int] = None,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        
        self.text_embedder = None
        if pos_embed_seq_len is not None:
            self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
            
        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
            
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.use_sp = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: Optional[Union[BlockMask, list[BlockMask]]] = None,
    ) -> torch.Tensor:
        if isinstance(attention_mask, list):
            attention_mask_0, attention_mask_1 = attention_mask
        elif isinstance(attention_mask, BlockMask) or attention_mask is None:
            attention_mask_0 = attention_mask
            attention_mask_1 = attention_mask
        else:
            raise TypeError(f"Expected attention_mask type: [list, BlockMask]. Got {type(attention_mask)}.")

        B, N, D, C = hidden_states.shape # hidden_states.shape: b (nt nh nw) (t h w) c
        hidden_states = rearrange(hidden_states, 'b n d c -> (b n) d c')
        scale_shift = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(scale_shift, 'b six c -> six b 1 c')
                
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        norm_hidden_states = rearrange(norm_hidden_states, '(b n) d c -> b n d c', b=B)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attention_mask_0
        )
        attn_output = rearrange(attn_output, 'b n d c -> (b n) d c')
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        norm_hidden_states = rearrange(norm_hidden_states, '(b n) d c -> b n d c', b=B)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask_1
        )
        attn_output = rearrange(attn_output, 'b n d c -> (b n) d c')
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        hidden_states = rearrange(hidden_states, '(b n) d c -> b n d c', b=B)
        return hidden_states
    
    def _set_sp_state(self, sp_group, sp_world_size, sp_rank, sp_src_rank):
        self.attn1.processor._set_sp_state(sp_group, sp_world_size, sp_rank, sp_src_rank)
        self.attn2.processor._set_sp_state(sp_group, sp_world_size, sp_rank, sp_src_rank)


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        attention_type: Optional[str] = 'slide_chunk_all_block',
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self.use_sp = False
        self._attention_map = {
            'full':                        partial(self._full_attention_mask),
            'slide_chunk':                 partial(self._sliding_chunk_attention_mask),
            'slide_chunk_6':               partial(self._sliding_chunk_attention_mask, num_slide_mask=6),
            'slide_chunk_all':             partial(self._sliding_chunk_attention_mask, num_slide_mask=len(self.blocks)),
            'slide_chunk_half_block':      partial(self._sliding_chunk_attention_mask, num_slide_mask=len(self.blocks) // 2, single_slide_per_block=False),
            'slide_chunk_all_block':       partial(self._sliding_chunk_attention_mask, num_slide_mask=len(self.blocks), single_slide_per_block=False),
            'slide_chunk_all_block_2x1x1': partial(self._sliding_chunk_attention_mask, window_T=2, num_slide_mask=len(self.blocks), single_slide_per_block=False),
            'slide_chunk_all_block_2x0x0': partial(self._sliding_chunk_attention_mask, window_T=2, window_H=0, window_W=0, num_slide_mask=len(self.blocks), single_slide_per_block=False),
            'slide_chunk_bottleneck':      partial(self._slide_chunk_bottlenect_attention_mask),
            'swin':                        partial(self._swin_attention_mask),
            'slide_window_4x16x16':        partial(self._sliding_window_attention_mask),
        }
        self.set_attention_type(attention_type)
        
    def _set_sp_state(self, sp_group, sp_world_size, sp_rank, sp_src_rank):
        self.use_sp = True
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.sp_rank = sp_rank
        self.sp_src_rank = sp_src_rank
        
    #----------------------------------------------------------------------
    # Attention.

    def set_attention_type(self, attention_type: str):
        assert attention_type in self._attention_map, \
            f"Expected attention_type: {self._attention_map.keys()}. Got {attention_type}."
        self._attention_type = attention_type
        
    def prepare_attention_mask(self, nT, nH, nW, T, H, W, device):
        return self._attention_map[self._attention_type](nT, nH, nW, T, H, W, device)

    @property
    def attention_type(self):
        return self._attention_type
    
    def _full_attention_mask(self, nT, nH, nW, T, H, W, device):
        return [None] * len(self.blocks)
        
    @lru_cache
    def _swin_attention_mask(self, nT, nH, nW, T, H, W, device):
        # flatten dims: (nt nh nw t h w)
        S = nT * T * nH * H * nW * W
        num_chunks = nT * nH * nW
        
        # Calculate padding
        BLOCK_SIZE = 128
        pad_len = (BLOCK_SIZE - S % BLOCK_SIZE) % BLOCK_SIZE

        id_to_chunk = torch.arange(num_chunks, dtype=torch.long, device=device)
        id_to_chunk = id_to_chunk.reshape(nT, nH, nW, 1, 1, 1)
        id_to_chunk = id_to_chunk.expand(nT, nH, nW, T, H, W)
        # Correct flatten order
        id_to_chunk = rearrange(id_to_chunk, 'nt nh nw t h w -> (nt nh nw t h w)')

        # Pad for safety
        if pad_len > 0:
            # Pad with -1 or distinct value to avoid accidental matching, though 0 is usually fine if logic holds
            id_to_chunk_padded = F.pad(id_to_chunk, (0, pad_len), value=-1)
        else:
            id_to_chunk_padded = id_to_chunk

        def chunk_mask_mod(b, h, q_idx, kv_idx):
            return id_to_chunk_padded[q_idx] == id_to_chunk_padded[kv_idx]

        chunk_mask = create_block_mask(chunk_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)

        # Pre-calculate shifted chunk IDs for swin attention.
        id_to_swin = rearrange(
            id_to_chunk, '(nt nh nw t h w) -> (nt t) (nh h) (nw w)',
            nt=nT, nh=nH, nw=nW, t=T, h=H, w=W
        )
        T_shift, H_shift, W_shift = T // 2, H // 2, W // 2
        id_to_swin = torch.roll(id_to_swin, shifts=(-T_shift, -H_shift, -W_shift), dims=(0, 1, 2))
        id_to_swin = rearrange(
            id_to_swin, '(nt t) (nh h) (nw w) -> (nt nh nw t h w)',
            t=T, h=H, w=W
        )
        
        # Pad shifted version
        if pad_len > 0:
            id_to_swin_padded = F.pad(id_to_swin, (0, pad_len), value=-2)
        else:
            id_to_swin_padded = id_to_swin

        def swin_mask_mod(b, h, q_idx, kv_idx):
            return id_to_swin_padded[q_idx] == id_to_swin_padded[kv_idx]

        swin_mask = create_block_mask(swin_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)

        attention_masks = []
        for i in range(len(self.blocks)):
            if i % 2 == 0:
                attention_masks.append(chunk_mask)  # W-MSA
            else:
                attention_masks.append(swin_mask)   # SW-MSA
        return attention_masks
    
    @lru_cache
    def _sliding_chunk_attention_mask(
        self, nT, nH, nW, T, H, W, device,
        window_T=1, window_H=1, window_W=1, 
        num_slide_mask=3, single_slide_per_block=True,
    ):
        # flatten dims: (nt nh nw t h w)
        S = nT * T * nH * H * nW * W 
        BLOCK_SIZE = 128
        pad_len = (BLOCK_SIZE - S % BLOCK_SIZE) % BLOCK_SIZE

        id_to_nt = torch.arange(nT, device=device).view(-1, 1, 1, 1, 1, 1)
        id_to_nh = torch.arange(nH, device=device).view(1, -1, 1, 1, 1, 1)
        id_to_nw = torch.arange(nW, device=device).view(1, 1, -1, 1, 1, 1)
        id_to_nt = id_to_nt.expand(nT, nH, nW, T, H, W).flatten()
        id_to_nh = id_to_nh.expand(nT, nH, nW, T, H, W).flatten()
        id_to_nw = id_to_nw.expand(nT, nH, nW, T, H, W).flatten()
        
        if pad_len > 0:
            id_to_nt = F.pad(id_to_nt, (0, pad_len))
            id_to_nh = F.pad(id_to_nh, (0, pad_len))
            id_to_nw = F.pad(id_to_nw, (0, pad_len))
        
        def sliding_window_mask_mod(b, h, q_idx, kv_idx):
            q_nt, q_nh, q_nw = id_to_nt[q_idx], id_to_nh[q_idx], id_to_nw[q_idx]
            kv_nt, kv_nh, kv_nw = id_to_nt[kv_idx], id_to_nh[kv_idx], id_to_nw[kv_idx]
            return (
                (torch.abs(q_nt - kv_nt) <= window_T) &
                (torch.abs(q_nh - kv_nh) <= window_H) &
                (torch.abs(q_nw - kv_nw) <= window_W)
            )
        
        def chunk_mask_mod(b, h, q_idx, kv_idx):
            q_nt, q_nh, q_nw = id_to_nt[q_idx], id_to_nh[q_idx], id_to_nw[q_idx]
            kv_nt, kv_nh, kv_nw = id_to_nt[kv_idx], id_to_nh[kv_idx], id_to_nw[kv_idx]
            return (
                (q_nt == kv_nt) &
                (q_nh == kv_nh) &
                (q_nw == kv_nw)
            )
        
        # S = nT * T * nH * H * nW * W 
        slide_mask = create_block_mask(sliding_window_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        chunk_mask = create_block_mask(chunk_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        masks = []
        for _ in range(len(self.blocks) % num_slide_mask):
            masks.append(chunk_mask)
        stride = len(self.blocks) // num_slide_mask
        for i in range(1, stride * num_slide_mask + 1):
            if i % stride == 0:
                if single_slide_per_block:
                    masks.append([chunk_mask, slide_mask])
                else:
                    masks.append(slide_mask)
            else:
                masks.append(chunk_mask)
        return masks
    
    @lru_cache
    def _slide_chunk_bottlenect_attention_mask(
        self, nT, nH, nW, T, H, W, device,
        window_T=2, window_H=8, window_W=8,
        bottleneck_ratio=0.8, window_T_bn=1, window_H_bn=1, window_W_bn=1,
    ):
        # flatten dims: (nt nh nw t h w)
        id_to_nt = torch.arange(nT, device=device).view(-1, 1, 1, 1, 1, 1)
        id_to_nh = torch.arange(nH, device=device).view(1, -1, 1, 1, 1, 1)
        id_to_nw = torch.arange(nW, device=device).view(1, 1, -1, 1, 1, 1)
        id_to_nt = id_to_nt.expand(nT, nH, nW, T, H, W).flatten()
        id_to_nh = id_to_nh.expand(nT, nH, nW, T, H, W).flatten()
        id_to_nw = id_to_nw.expand(nT, nH, nW, T, H, W).flatten()
        
        def sliding_window_mask_mod(b, h, q_idx, kv_idx, window_T, window_H, window_W):
            q_nt, q_nh, q_nw = id_to_nt[q_idx], id_to_nh[q_idx], id_to_nw[q_idx]
            kv_nt, kv_nh, kv_nw = id_to_nt[kv_idx], id_to_nh[kv_idx], id_to_nw[kv_idx]
            return (
                (torch.abs(q_nt - kv_nt) <= window_T) &
                (torch.abs(q_nh - kv_nh) <= window_H) &
                (torch.abs(q_nw - kv_nw) <= window_W)
            )
        
        S = nT * T * nH * H * nW * W 
        slide_mask = create_block_mask(
            partial(sliding_window_mask_mod, window_T=window_T, window_H=window_H, window_W=window_W), 
            B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True
        )
        slide_mask_bn = create_block_mask(
            partial(sliding_window_mask_mod, window_T=window_T_bn, window_H=window_H_bn, window_W=window_W_bn), 
            B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True
        )
        
        num_bn_layers = int(len(self.blocks) * bottleneck_ratio)
        num_out_layers = (len(self.blocks) - num_bn_layers) // 2
        num_in_layers = len(self.blocks) - num_bn_layers - num_out_layers
        masks = [slide_mask] * num_in_layers + [slide_mask_bn] * num_bn_layers + [slide_mask] * num_out_layers
        return masks
    
    @lru_cache
    def _sliding_window_attention_mask(
        self, nT, nH, nW, T, H, W, device,
        window_T=4, window_H=16, window_W=16,
    ):
        # flatten dims: (nt nh nw t h w)
        id_to_t = torch.arange(nT * T, device=device)
        id_to_h = torch.arange(nH * H, device=device)
        id_to_w = torch.arange(nW * W, device=device)
        id_to_t = repeat(id_to_t, '(nt t) -> (nt nh nw t h w)', nt=nT, nh=nH, nw=nW, t=T, h=H, w=W)
        id_to_h = repeat(id_to_h, '(nh h) -> (nt nh nw t h w)', nt=nT, nh=nH, nw=nW, t=T, h=H, w=W)
        id_to_w = repeat(id_to_w, '(nw w) -> (nt nh nw t h w)', nt=nT, nh=nH, nw=nW, t=T, h=H, w=W)

        def sliding_window_mask_mod(b, h, q_idx, kv_idx):
            q_t, q_h, q_w = id_to_t[q_idx], id_to_h[q_idx], id_to_w[q_idx]
            kv_t, kv_h, kv_w = id_to_t[kv_idx], id_to_h[kv_idx], id_to_w[kv_idx]
            return (
                (torch.abs(q_t - kv_t) <= window_T) &
                (torch.abs(q_h - kv_h) <= window_H) &
                (torch.abs(q_w - kv_w) <= window_W)
            )
                
        S = nT * T * nH * H * nW * W 
        slide_mask = create_block_mask(sliding_window_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        masks = []
        for _ in range(len(self.blocks)):
            masks.append(slide_mask)
        return masks

    #----------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        B, nT, nH, nW = hidden_states.shape[:4]
        p_t, p_h, p_w = self.config.patch_size
        
        # 1. Patch embedding.
        hidden_states = rearrange(
            hidden_states, "b nt nh nw c t h w -> b c (nt t) (nh h) (nw w)",
        )
        rotary_emb = self.rope(hidden_states)
        hidden_states = rearrange(
            hidden_states, "b c (nt t) (nh h) (nw w) -> (b nt nh nw) c t h w",
            nt=nT, nh=nH, nw=nW,
        )
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = rearrange(
            hidden_states, "(b nt nh nw) c t h w -> b nt nh nw c t h w",
            nt=nT, nh=nH, nw=nW,
        )
        B, nT, nH, nW, C, T, H, W = hidden_states.shape
        hidden_states = rearrange(
            hidden_states, "b nt nh nw c t h w -> b (nt nh nw) (t h w) c",
        )
        rotary_emb = rearrange(
            rotary_emb, '1 1 (nt t nh h nw w) c -> (nt nh nw) (t h w) c',
            nt=nT, nh=nH, nw=nW, t=T, h=H, w=W
        )

        # 2. Condition embedding.
        timestep = rearrange(timestep, "b nt nh nw -> (b nt nh nw)")
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        ) # temb: (b nt nh nw) c
        assert encoder_hidden_states_image is None, "encoder_hidden_states_image currently are not supported."
        assert encoder_hidden_states is None, "encoder_hidden_states currently are not supported."
        timestep_proj = timestep_proj.unflatten(1, (6, -1)) # (b nt nh nw) 6 c
        
        if self.use_sp:
            hidden_states = hidden_states.chunk(self.sp_world_size, dim=-2)[self.sp_rank]
            rotary_emb = rotary_emb.chunk(self.sp_world_size, dim=-2)[self.sp_rank]
        
        # 3. Transformer blocks
        attention_masks = self.prepare_attention_mask(nT, nH, nW, T, H, W, hidden_states.device)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block, attention_mask in zip(self.blocks, attention_masks):
                block_fn = partial(block, attention_mask=attention_mask)
                hidden_states = self._gradient_checkpointing_func(
                    block_fn, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                )
        else:
            for block, attention_mask in zip(self.blocks, attention_masks):
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, 
                    attention_mask=attention_mask
                )
        hidden_states = rearrange(hidden_states, 'b n d c -> (b n) d c')

        # 4. Output norm, projection & unpatchify
        scale_shift_table = rearrange(self.scale_shift_table, "1 two c -> two 1 1 c")
        shift, scale = scale_shift_table + temb.unsqueeze(1) # (b nt nh nw) 1 c

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        if self.use_sp:
            hidden_states = gather_forward(hidden_states, dim=1, group=self.sp_group)
        hidden_states = rearrange(
            hidden_states, "(b nt nh nw) (t h w) (p_t p_h p_w c) -> b nt nh nw c (t p_t) (h p_h) (w p_w)", 
            b=B, nt=nT, nh=nH, nw=nW, t=T, h=H, w=W, p_t=p_t, p_h=p_h, p_w=p_w,
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)