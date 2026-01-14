import argparse
import logging
import os
from pathlib import Path
from PIL import Image

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import torchvision
import diffusers
from einops import rearrange

from training.util import print_module_summary
from training.distributed import parallel_state_sp as sp_state
from training.models import WanTransformer3DModel
from training.models.precond import (
    Precond,
    Wan2_1SpatialTiledEncoder3D,
    Wan2_1SpatialTiledConditionEncoder3D,
    Wan2_1SpatialTiledConditionEncoder3Dv2,
    MaskSpatialTiledEncoder3D,
)
from generate import sample_skip_concat, VideoFrameLazyReader

logger = get_logger(__name__, log_level="INFO")
            
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
            model.set_sp_state(sp_group, sp_world_size, sp_rank, sp_src_rank)

        if accelerator.is_main_process:
            print_module_summary(model, 2)

        model.to(device=device, dtype=weight_dtype).eval()
    
    #------------------------------------------------------------------------------------------------------
    # Main generation loop.

    logger.info("***** Running generation *****")
    logger.info(f"  Sampling steps = {args.sampling_steps}")
    logger.info(f"  Spatial upsampling factor = {args.spatial_sf}")
    logger.info(f"  Temporal upsampling factor = {args.temporal_sf}")

    data_dir = Path(args.data_dir)
    gt_path = []
    gt_path.extend((data_dir / "test" / "Type1").glob("*"))
    gt_path.extend((data_dir / "test" / "Type2").glob("*"))
    gt_path.extend((data_dir / "test" / "Type3").glob("*"))

    def _resize_img(x):
        if args.task == "4k":
            return x
        x = x.float().unsqueeze(0)
        if args.task == "2k":
            x = F.interpolate(x, size=(1080, 2048), mode="area")
        elif args.task == "1k":
            x = F.interpolate(x, size=(540, 1024), mode="area")
        else:
            raise ValueError(f"args.task = {args.task}")
        x = x.squeeze(0).clip(0, 255).byte()
        return x
    
    for i in tqdm(
        range(dp_rank, len(gt_path), dp_size), desc="Total progress", disable=(sp_rank != 0)
    ):
        # Read data.
        gt_seq = VideoFrameLazyReader(gt_path[i], dimension_order='NCHW')
        gt_files = gt_seq._files
        gt_seq = torch.stack([_resize_img(gt_seq[t]) for t in range(len(gt_seq))])
        lq_seq = gt_seq[::args.temporal_sf]

        # Create sampler.
        sample_fn = sample_skip_concat(
            lq_seq,
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
                lq_list.append(lq_uint8.detach().cpu())
                pred_list.append(pred_uint8.detach().cpu())
            if args.max_chunks is not None and j + 1 >= args.max_chunks:
                logger.info(f"Reached max chunks: {args.max_chunks}. Stopping generation.")
                break     

        if sp_rank == 0:
            output_dir = Path(args.output_dir) / gt_path[i].relative_to(data_dir)
            os.makedirs(output_dir, exist_ok=True)
            codec_kwargs = {
                'video_codec': "libx264",
                'options': {'crf': '10', 'pix_fmt': 'yuv420p'}
            }
            lq_uint8 = rearrange(torch.cat(lq_list, dim=0), 't c h w -> t h w c')
            pred_uint8 = rearrange(torch.cat(pred_list, dim=0), 't c h w -> t h w c')[:len(gt_seq)]
            gt_uint8 = rearrange(gt_seq, 't c h w -> t h w c')

            # Save videos.
            torchvision.io.write_video(output_dir / 'pred.mp4', pred_uint8, fps=args.fps, **codec_kwargs)
            torchvision.io.write_video(output_dir / 'gt.mp4', gt_uint8, fps=args.fps, **codec_kwargs)
            if args.save_lq:
                torchvision.io.write_video(output_dir / 'lq.mp4', lq_uint8, fps=args.fps / args.temporal_sf, **codec_kwargs)

            # Save frames.
            img_dir = Path(args.output_dir) / "imgs"
            for j, file in enumerate(gt_files):
                save_dir = "--".join((Path(file).parent.relative_to(data_dir)).parts)
                fname = Path(file).name
                pred_save_dir = img_dir / "pred" / save_dir
                gt_save_dir = img_dir / "gt" / save_dir
                os.makedirs(pred_save_dir, exist_ok=True)
                os.makedirs(gt_save_dir, exist_ok=True)
                Image.fromarray(pred_uint8[j].detach().cpu().numpy()).save(pred_save_dir / fname)
                Image.fromarray(gt_uint8[j].detach().cpu().numpy()).save(gt_save_dir / fname)
            logger.info(f"Results saved to {output_dir}")
            
    accelerator.end_training()
    
#----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--data_dir",                  type=str,              required=True,           help="Path to SNU_FILM")
    parser.add_argument("--task",                      type=str,              default="1k",            choices=["1k", "2k", "4k"])
    parser.add_argument("--fps",                       type=float,            default=30,              help="Frames per second of the video.")

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
    parser.add_argument("--seed",                      type=int,              default=42,              help="Random seed.")
    parser.add_argument("--batch_vae",                 type=int,              default=16,              help="[Deprecated] VAE Batch size per device. Use --vae_batch_size instead.")
    parser.add_argument("--sampling_steps",            type=int,              default=100,             help="Number of denoising steps.")
    parser.add_argument("--t_cond",                    type=float,            default=0.1,             help="t_cond for AR generation.")
    parser.add_argument("--t_shift",                   type=float,            default=1.0,             help="Timestep shift for high resolution.")
    parser.add_argument("--upsample_lq",               type=str,              default='trilinear',     choices=['nearest', 'trilinear', 'pad'], help="Temporal upsample mode for LQ: interpolate or zero-pad frames.")
    parser.add_argument("--spatial_sf",                type=int,              default=1,               help="Spatial upsampling factor.")
    parser.add_argument("--temporal_sf",               type=int,              default=4,               help="Temporal upsampling factor.")
    parser.add_argument("--output_dir",                type=str,              default=None,            help="Path to output directory for inference results.")
    parser.add_argument("--max_chunks",                type=int,              default=None,            help="Maximum number of chunks to process during inference.")
    parser.add_argument("--sp_size",                   type=int,              default=1,               help="Sequence parallel size.")
    parser.add_argument("--save_lq",                   action="store_true",   default=False,           help="Save the low-quality and upsampled videos.")
    
    # FID argument.
    parser.add_argument("--compute_fid",               action="store_true",                   help="Compute FID over generated vs GT frames.")
    parser.add_argument("--fid_dims",                  type=int,             default=2048,            help="Dimensionality of Inception features to use for FID (64, 192, 768, 2048).")
    parser.add_argument("--fid_batch_size",            type=int,             default=32,              help="Batch size for FID feature extraction.")
    parser.add_argument("--fid_num_workers",           type=int,             default=4,               help="Number of workers for FID dataloader.")
    
    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)