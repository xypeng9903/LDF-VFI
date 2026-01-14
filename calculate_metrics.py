"""Calculate video interpolation metrics across folder trees.

Given two root directories (pred_dir, gt_dir) each containing multiple video
folders (either nested or flattened with `--` delimiters), this script
computes per-video and global PSNR / SSIM / LPIPS.

Flattened directory convention (for both GT and Pred):
    GT scene dir name:   test--Type1--SceneA
    Pred scene dir name: test--Type1--SceneA (same name under pred root)

Usage example:
    python calculate_metrics.py \
        --pred-dir RIFE-X4K1000FPS-8x \
        --gt-dir /datasets/X4K1000FPS \
        --metrics psnr,ssim,lpips \
        --skip-step 4 \
        --batch-size 8 --num-workers 4 \
        --output-json results.json --output-csv per_video.csv

Notes:
  * Assumes image filenames align between pred & gt (e.g. 0000.png ...).
  * Missing prediction frames are skipped with a warning.
  * Accelerate is used for multi-GPU; only main process writes outputs.
"""

import argparse
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from accelerate import Accelerator
import torch.distributed as dist
from tqdm.auto import tqdm

from metrics.cal_metrics import CalMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Calculate interpolation metrics (PSNR, SSIM, LPIPS)")
    p.add_argument('--pred-dir', type=str, required=True, help='Prediction root directory')
    p.add_argument('--gt-dir', type=str, required=True, help='Ground truth root directory')
    p.add_argument('--skip-step', type=int, default=0, help='Order-based evaluation stride S (e.g. 4/8/16). Treat frames in folder order; assume key/context frames every S positions (0,S,2S,...). For each full block [b, b+S] that exists, only evaluate intermediate frames b+1..b+S-1. Discard the last incomplete tail block. Set 0 to disable and evaluate all frames.')
    p.add_argument('--image-ext', type=str, default='png', help='Image extension to scan')
    p.add_argument('--metrics', type=str, default='psnr,ssim,lpips', help='Comma list: psnr,ssim,lpips')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--resize', type=str, default=None, help='Optional resize HxW for metrics (e.g. 256x256)')
    p.add_argument('--output-json', type=str, default=None)
    p.add_argument('--output-csv', type=str, default=None)
    p.add_argument('--max-videos', type=int, default=None, help='Optional cap for debugging')
    return p.parse_args()


def list_flattened_scene_dirs(root: str) -> List[str]:
    """List immediate subdirectories that contain at least one image file.

    Assumes both GT and Pred roots contain flattened scene directories as direct children.
    """
    dirs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        # Has at least one image inside
        try:
            files = os.listdir(p)
        except Exception:
            continue
        if any(re.search(r'\.(png|jpg|jpeg)$', f.lower()) for f in files):
            dirs.append(p)
    return dirs


def basename(p: str) -> str:
    return os.path.basename(os.path.normpath(p))


def map_pred_video_dirs_flat(gt_root: str, pred_root: str) -> Dict[str, str]:
    """Map GT scene dirs to Pred scene dirs by matching flattened names (direct children)."""
    gt_dirs = list_flattened_scene_dirs(gt_root)
    pred_dirs = list_flattened_scene_dirs(pred_root)
    gt_index = {basename(p): p for p in gt_dirs}
    pred_index = {basename(p): p for p in pred_dirs}
    names = sorted(set(gt_index.keys()) & set(pred_index.keys()))
    mapping = {gt_index[n]: pred_index[n] for n in names}
    return mapping


def load_image(path: str, transform, device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    if transform:
        img = transform(img)
    return img.to(device)


### NOTE (skip-step semantics updated)
# We no longer rely on parsing numbers from filenames to decide which frames to skip.
# Instead, we use the ORDER of frames inside the directory (lexicographically sorted
# by filename, same as before) and interpret --skip-step = S as:
#   Key frames at positions 0, S, 2S, ... (these are the original input/context frames).
#   For each COMPLETE block of length S between key frame b and b+S (i.e. b+S < N), we
#   evaluate only the intermediate frames: indices b+1 .. b+S-1. The last incomplete
#   block (if total frames N does not end exactly on a key frame) is discarded entirely
#   for evaluation. This mirrors interpolation scenarios (e.g. 4x: only frames 1,2,3
#   between 0 and 4 are valid if frame 4 exists; frames after the last full block are
#   ignored because there is no future context frame).


def compute_image_metrics(cal: CalMetrics, frame_pairs: List[Tuple[str, str]], batch_size: int, transform, accelerator: Accelerator, metrics: List[str], desc: str | None = None) -> Dict[str, float]:
    device = accelerator.device
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    pbar = None
    if accelerator.is_local_main_process:
        pbar = tqdm(total=len(frame_pairs), desc=desc or 'video', leave=False)
    # Load in batches
    for i in range(0, len(frame_pairs), batch_size):
        batch = frame_pairs[i:i+batch_size]
        preds = []
        gts = []
        for p_path, g_path in batch:
            pred = load_image(p_path, transform, device)
            gt = load_image(g_path, transform, device)
            preds.append(pred)
            gts.append(gt)
        pred_t = torch.stack(preds, 0)
        gt_t = torch.stack(gts, 0)
        # Range [0,1]
        with torch.no_grad():
            if 'psnr' in metrics:
                psnr_vals.append(cal.cal_psnr(pred_t, gt_t).detach())
            if 'ssim' in metrics:
                ssim_vals.append(cal.cal_ssim(pred_t, gt_t).detach())
            if 'lpips' in metrics:
                lpips_vals.append(cal.cal_lpips(pred_t, gt_t).detach())
        if pbar is not None:
            pbar.update(len(batch))
    if pbar is not None:
        pbar.close()
    
    # Aggregate across processes
    out = {}
    
    def reduce_metric(vals_list):
        if vals_list:
            flat = torch.cat([v.view(-1) for v in vals_list])
            local_sum = flat.sum()
            local_count = torch.tensor(flat.numel(), device=device, dtype=torch.long)
        else:
            local_sum = torch.tensor(0.0, device=device)
            local_count = torch.tensor(0, device=device, dtype=torch.long)
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            
        if local_count.item() > 0:
            return (local_sum / local_count).item()
        else:
            return float('nan')

    if 'psnr' in metrics:
        out['psnr'] = reduce_metric(psnr_vals)
    if 'ssim' in metrics:
        out['ssim'] = reduce_metric(ssim_vals)
    if 'lpips' in metrics:
        out['lpips'] = reduce_metric(lpips_vals)
    return out


def collect_frames(dir_path: str, ext: str) -> List[str]:
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.' + ext.lower())])
    return [os.path.join(dir_path, f) for f in files]


def match_frames(pred_dir: str, gt_dir: str, ext: str, skip_step: int = 0) -> List[Tuple[str, str]]:
    """Match prediction frames to GT frames respecting order-based skip semantics.

    If skip_step == 0: evaluate all frames that exist in predictions.
    Else: treat frames as an ordered list; define key frames at indices k*S (0-based),
    and only evaluate intermediate frames within COMPLETE blocks where (k+1)*S < N.
    E.g. frames 0 1 2 3 4 5 6 with S=4 -> only evaluate 1,2,3 (since 4 exists); 5,6
    are dropped because there is no frame 8 to close the block.
    """
    pred_frames = collect_frames(pred_dir, ext)
    gt_frames = collect_frames(gt_dir, ext)
    pred_map = {Path(p).name: p for p in pred_frames}
    pairs: List[Tuple[str, str]] = []
    missing = 0
    N = len(gt_frames)

    if skip_step and skip_step > 0:
        S = skip_step
        # Iterate over block starts (key frame positions). A block is valid if b+S < N.
        for b in range(0, N, S):
            end = b + S
            if end >= N:  # incomplete tail block -> discard entirely
                break
            # Intermediate frames in the full block
            for i in range(b + 1, end):
                g = gt_frames[i]
                name = Path(g).name
                if name in pred_map:
                    pairs.append((pred_map[name], g))
                else:
                    missing += 1
    else:
        # No skipping; include every frame that has a prediction counterpart.
        for g in gt_frames:
            name = Path(g).name
            if name in pred_map:
                pairs.append((pred_map[name], g))
            else:
                missing += 1

    if missing > 0:
        print(f'[WARN] {missing} frames missing in pred for video {gt_dir}')
    return pairs


def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_local_main_process:
        print('Args:', args)
        # Quick device summary to clarify GPU usage
        try:
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(device) if cuda_ok else 'CPU'
        except Exception:
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU'
        print(f'Device: {device} | CUDA available: {cuda_ok} | GPU: {gpu_name}')

    metrics_requested = [m.strip() for m in args.metrics.split(',') if m.strip()]
    resize_transform = None
    if args.resize:
        h, w = map(int, args.resize.lower().split('x'))
        resize_transform = T.Compose([T.ToTensor(), T.Resize((h, w))])
    else:
        resize_transform = T.ToTensor()

    with accelerator.main_process_first():
        mapping = map_pred_video_dirs_flat(args.gt_dir, args.pred_dir)
        gt_video_dirs = sorted(mapping.keys())
        if args.max_videos:
            gt_video_dirs = gt_video_dirs[:args.max_videos]
            mapping = {k: mapping[k] for k in gt_video_dirs}
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    # Shard videos across processes for data-parallel evaluation
    # gt_video_dirs_local = gt_video_dirs[rank::world_size] if world_size > 1 else gt_video_dirs
    gt_video_dirs_local = gt_video_dirs
    mapping_local = {k: mapping[k] for k in gt_video_dirs_local}
    if accelerator.is_local_main_process:
        print(f'Found {len(gt_video_dirs)} GT videos, {len(mapping)} with predictions.')
        if world_size > 1:
            print(f'Data-parallel over images: world_size={world_size}, rank={rank}')

    cal = CalMetrics()
    per_video_rows_local = []

    # Per-video metrics with global progress bar
    video_iter = gt_video_dirs_local
    if accelerator.is_local_main_process:
        from tqdm.auto import tqdm as _tqdm
        video_iter = _tqdm(gt_video_dirs_local, desc='Videos', dynamic_ncols=True)
    for gt_dir in video_iter:
        pred_dir = mapping_local[gt_dir]
        frame_pairs = match_frames(pred_dir, gt_dir, args.image_ext, skip_step=args.skip_step)
        if not frame_pairs:
            continue
        
        # Shard frames across processes
        frame_pairs_local = frame_pairs[rank::world_size]
        
        stats = compute_image_metrics(
            cal,
            frame_pairs_local,
            args.batch_size,
            resize_transform,
            accelerator,
            metrics_requested,
            desc=os.path.basename(gt_dir),
        )
        row = {'video': os.path.basename(gt_dir)}
        for k, v in stats.items():
            row[k] = v
        
        if accelerator.is_local_main_process:
            per_video_rows_local.append(row)
            print(f'[Video] {row}')

    # Gather per-video rows from all processes
    accelerator.wait_for_everyone()
    merged_rows = None
    if accelerator.is_local_main_process:
        merged_rows = per_video_rows_local

    # Compute global stats on main from merged rows
    global_stats = {m: float('nan') for m in metrics_requested}
    if accelerator.is_local_main_process and merged_rows is not None:
        for m in metrics_requested:
            vals = [r[m] for r in merged_rows if m in r]
            global_stats[m] = (sum(vals) / len(vals)) if len(vals) > 0 else float('nan')

    result = {
        'num_videos': (len(merged_rows) if accelerator.is_local_main_process and merged_rows is not None else len(per_video_rows_local)),
        'per_video': (merged_rows if accelerator.is_local_main_process and merged_rows is not None else per_video_rows_local),
        'global': global_stats,
    }

    if accelerator.is_local_main_process:
        print('[Global]', json.dumps(result['global'], indent=2))
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'Wrote JSON: {args.output_json}')
        if args.output_csv:
            header = ['video'] + metrics_requested
            write_csv(args.output_csv, result['per_video'], header)
            print(f'Wrote CSV: {args.output_csv}')


if __name__ == '__main__':
    main()