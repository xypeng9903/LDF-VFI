# Modifed from:
# https://github.com/bbldCVer/EDEN/blob/11789cd8288127c9f34f1c40f57b317e767fbf43/src/utils/cal_metrics.py

import math
import torch
from torch.nn.functional import conv3d, pad
from lpips import LPIPS
from .flolpips.flolpips import Flolpips


class PSNR:
    def __call__(self, pred, gt):
        b = gt.shape[0]
        se = (pred - gt) ** 2
        mse = torch.mean(se.reshape(b, -1), dim=1)
        return 10 * torch.log10((255. ** 2) / mse)


# based on UPRNet repo.
# ssim_matlab.
# https://github.com/JHLew/MoMo/blob/main/evaluation/metrics.py
class SSIM:
    def __init__(self, window_size=11, window=None, size_average=False, full=False, val_range=None) -> None:
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window_3d(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
        window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous()
        return window

    def __call__(self, img1, img2):
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range

        padd = 0
        (_, _, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window_3d(real_size, channel=1).to(img1.device)
            # Channel is set to 1 since we consider color images as volumetric images
        else:
            window = self.window

        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)

        mu1 = conv3d(pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
        mu2 = conv3d(pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv3d(pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
        sigma2_sq = conv3d(pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
        sigma12 = conv3d(pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1).mean(1)

        if self.full:
            return ret, cs
        return ret


class CalMetrics:
    def __init__(self):
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS().eval()
        self.flolpips = Flolpips().eval()
        self.scaler = 255.0

    def to_uint8(self, img):
        img = img * self.scaler
        img = img.round()
        img = torch.clamp(img, 0, 255)
        return img

    def quantize(self, img):
        img = self.to_uint8(img)
        return img / self.scaler

    def cal_lpips(self, pred, gt):
        """ Note: pred and gt have range [0,1]"""
        self.lpips = self.lpips.to(pred.device)
        pred = self.quantize(pred)
        lpips_value = self.lpips(pred, gt, normalize=True)
        lpips_value = lpips_value.sum() / pred.shape[0]
        return lpips_value

    def cal_psnr(self, pred, gt):
        """ Note: pred and gt have range [0,1]"""
        pred, gt = self.to_uint8(pred), self.to_uint8(gt)
        psnr_value = self.psnr(pred, gt)
        psnr_value = psnr_value.sum() / pred.shape[0]
        return torch.tensor(psnr_value).to(pred.device)

    def cal_ssim(self, pred, gt):
        """ Note: pred and gt have range [0,1]"""
        pred = self.quantize(pred)
        ssim_value = self.ssim(pred, gt)
        ssim_value = ssim_value.sum() / pred.shape[0]
        return ssim_value

    def cal_flolpips(self, pred, gt, i0, i1):
        """ Note: pred and gt have range [0,1]"""
        self.flolpips = self.flolpips.to(pred.device)
        flolpips_value = self.flolpips(i0, i1, pred, gt)
        flolpips_value = flolpips_value.sum() / pred.shape[0]
        return flolpips_value