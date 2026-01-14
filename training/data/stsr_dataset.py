from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
import torch
import math
import torch.nn.functional as F
from torchvision.transforms.functional import vflip, hflip
import random

    
class TemporalDegradation:
    def __init__(self, temporal_sf: list):
        self.temporal_sf = temporal_sf

    def apply(self, x: torch.Tensor):
        # input: x (tchw)
        # return: x (tchw), msk (t)
        temporal_sf = random.choice(self.temporal_sf)
        msk = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        high = min(temporal_sf - 1, msk.shape[0] - 1)
        start = random.randint(0, high)
        msk[start : : temporal_sf] = True
        x = x[msk]
        return x, msk
    

def upsample_temporal(x: torch.Tensor, msk: torch.Tensor, mode: str):
    C, H, W = x.shape[-3:]
    T = msk.shape[0]

    if mode == 'pad':
        tmp = torch.zeros((T, C, H, W), device=x.device, dtype=x.dtype)
        tmp[msk] = x
        x = tmp
    elif mode == 'nearest':
        kept_indices = torch.where(msk)[0]
        positions = torch.arange(T, device=x.device)
        distances = (positions.view(-1, 1) - kept_indices.view(1, -1)).abs()
        nearest_kept = distances.argmin(dim=1)  # (T,)
        x = x[nearest_kept]
    else:
        raise NotImplementedError(f'Unsupported mode: {mode}')
    return x


class VideoDataset(Dataset):
    pn_to_resolution = {
        '0.06M': [(256, 256)],
        '0.25M': [(512, 512)],
        '1M':    [(1024, 1024), (720, 1280), (1280, 720)],
        '4M':    [(2048, 2048), (1280, 2560), (2560, 1280), (1024, 4096), (4096, 1024)],
    }
    
    def __init__(self, meta_data,
        num_frames  = 60, 
        pn          = "0.25M", 
        dynamic_res = False,
        no_bicubic  = True, 
        p_vflip     = 0, 
        p_hflip     = 0, 
        p_transpose = 0,
    ):        
        self.meta_data = meta_data
        
        self.num_frames = num_frames
        if dynamic_res:
            self.resolution = self.pn_to_resolution[pn]
        else:
            self.resolution = self.pn_to_resolution[pn][0:1]
        self.pn = pn
        self.dynamic_res = dynamic_res
        self.no_bicubic = no_bicubic
        self.p_vflip = p_vflip
        self.p_hflip = p_hflip
        self.p_transpose = p_transpose
        
    def __len__(self):
        return len(self.meta_data)
    
    def _getitem_fn(self, idx):
        # Read video.
        example = self.meta_data[idx].copy()
        gt_reader = VideoDecoder(example['video'], dimension_order='NCHW')
        input_T = self.num_frames
        j = torch.randint(0, len(gt_reader) - input_T + 1, (1,)).item()
        gt = gt_reader[j : j + input_T]
        assert gt.shape[0] == input_T, f"gt.shape[0] ({gt.shape[0]}) != input_T ({input_T})"
        _, _, h, w = gt.shape
        
        # Get GT.
        input_H, input_W = self.resolution[torch.randint(0, len(self.resolution), (1,)).item()]
        h_sf = input_H / h
        w_sf = input_W / w
        sf = max(h_sf, w_sf)
        if not self.no_bicubic or sf > 1:
            new_h, new_w = math.ceil(h * sf), math.ceil(w * sf)
            gt = F.interpolate(gt.float(), (new_h, new_w), mode='bicubic')
            gt = gt.to(dtype=torch.uint8)
        else:
            new_h, new_w = h, w     
        top = torch.randint(0, new_h - input_H + 1, (1,)).item()
        left = torch.randint(0, new_w - input_W + 1, (1,)).item()
        gt = gt[..., top: top + input_H, left: left + input_W]
        if torch.rand(1).item() < self.p_vflip: gt = vflip(gt)
        if torch.rand(1).item() < self.p_hflip: gt = hflip(gt)
        if torch.rand(1).item() < self.p_transpose: gt = gt.transpose(-2, -1)    
        example.update({
            'gt': gt.to(dtype=torch.uint8).contiguous(),
        })
        return example
    
    def __getitem__(self, idx):
        # return self._getitem_fn(idx) # for debugging
        n_tries = 10
        for _ in range(n_tries):
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                print(f"Error loading video {self.meta_data[idx]['video']}: {e}")
                idx = torch.randint(0, len(self.meta_data), (1,)).item()
        raise RuntimeError(f"Failed to load video after {n_tries} tries (last tried: {self.meta_data[idx]['video']})")