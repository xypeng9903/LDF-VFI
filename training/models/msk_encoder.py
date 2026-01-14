from torch import nn
import torch
from torch.nn import functional as F
from einops import rearrange    


class MaskEncoder(nn.Module):
    def __init__(self, spatial_compression_ratio=8):
        super().__init__()
        self.spatial_compression_ratio = spatial_compression_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, 1, T, H, W
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 按时间拆分为1、4、4、4....
        for i in range(iter_):
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :])
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :])
                out = torch.cat([out, out_], 2)
        return out
    
    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 4, 1, 1)
        assert x.shape[2] == 4
        x = F.interpolate(
            x,
            size=(
                x.shape[-3], 
                x.shape[-2] // self.spatial_compression_ratio, 
                x.shape[-1] // self.spatial_compression_ratio
            ), 
            mode='nearest'
        )
        x = rearrange(x, 'b 1 (t c) h w -> b c t h w', c=4)
        return x