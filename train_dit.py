import argparse
import logging
import os
import warnings
import json
from time import time
from copy import deepcopy
import re
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
import diffusers
from peft import LoraConfig
from einops import rearrange, repeat

import training.distributed.parallel_state_sp as sp_state


logger = get_logger(__name__, log_level="INFO")

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Data arguments
    parser.add_argument("--data",                      type=str,             required=True,           help="Path to the dataset on disk.")
    parser.add_argument("--spatial_opt",               type=str,             required=None,           help="Path to the spatial degradation options (YAML file).")
    parser.add_argument("--num_frames",                type=int,             default=60,              help="Number of frames of the video.")
    parser.add_argument("--pn",                        type=str,             default='0.25M',         choices=["0.06M", "0.25M", "1M", "4M"])
    parser.add_argument("--temporal_sf",               type=parse_int_list,  default="4",             help="temporal downsample scaling factor of the video.")
    parser.add_argument("--avg_frames",                type=str,             default="0,0",           help="Range of number of frames to average for temporal downsample.")
    parser.add_argument("--spatial_sf",                type=str,             default="1,2",           help="Spatial downsample scaling factor of the video.")
    parser.add_argument("--no_bicubic",                action="store_true",  default=False,           help="Whether to disable bicubic interpolation.")
    parser.add_argument("--p_vflip",                   type=float,           default=0,               help="Probability of vertical flipping.")
    parser.add_argument("--p_hflip",                   type=float,           default=0,               help="Probability of horizontal flipping.")
    parser.add_argument("--p_transpose",               type=float,           default=0,               help="Probability of transposing.")
    parser.add_argument("--dynamic_res",               action="store_true",  default=False,           help="Enable dynamic resolution sampling in dataset.")
    parser.add_argument("--temporal_upsample",         type=str,             default='nearest',       help="Temporal upsampling mode.")
    parser.add_argument("--subsample_gt",              type=str,             default="1",             help="Subsample factor for ground-truth video.")
    parser.add_argument("--pad_gt",                    action="store_true",  default=False,           help="Whether to pad the ground-truth video.")

    # Model arguments
    parser.add_argument("--model_path",                type=str,             default=None,            help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_config",              type=str,             default=None,            help="Path to model config.")
    parser.add_argument("--vae_channels",              type=int,             default=None,            help="Number of channels in the VAE.")
    parser.add_argument("--attention_type",            type=str,             default=None,            help="Attention type for the model.")
    parser.add_argument("--diffusion_forcing",         type=str,             default='all',           choices=['all', 'temporal'])
    parser.add_argument("--lora_rank",                 type=int,             default=0,               help="LoRA rank r. Set >0 to enable PEFT.")
    parser.add_argument("--lq_proj_hidden_dim",        type=int,             default=64,              help="Hidden dim for LQProjSpatialTiledEncoder3D.")

    # VAE arguments
    parser.add_argument("--vae_type",                  type=str,             default='wan2_1_cond')
    parser.add_argument("--vae_path",                  type=str,             default=None,            help="Path to VAE.")
    parser.add_argument("--vae_batch_size",            type=int,             default=2,               help="Per-device batch size used inside VAE encode/decode.")
    parser.add_argument("--tile_min_h",                type=int,             default=128,             help="Tile minimum height (pixels).")
    parser.add_argument("--tile_min_w",                type=int,             default=128,             help="Tile minimum width (pixels).")
    parser.add_argument("--tile_min_t",                type=int,             default=20,              help="Tile minimum temporal length (frames). For cosmos, must be divisible by 5.")
    parser.add_argument("--tile_stride_h",             type=int,             default=96,              help="Tile stride height (pixels).")
    parser.add_argument("--tile_stride_w",             type=int,             default=96,              help="Tile stride width (pixels).")
    parser.add_argument("--spatial_compression_ratio", type=int,             default=8,               help="Spatial compression ratio of the VAE.")
    parser.add_argument("--temporal_compression_ratio",type=int,             default=4,               help="Temporal compression ratio of the VAE.")

    # Training arguments
    parser.add_argument("--t_shift",                   type=float,           default=1.0,             help="Timestep shift for high resolution.")
    parser.add_argument("--start_step",                type=int,             default=None,            help="Step to start training from.")
    parser.add_argument("--seed",                      type=int,             default=42,              help="Random seed for initialization.")
    parser.add_argument("--output_dir",                type=str,             default="ddpm-model-64", help="Output directory for model and checkpoints.")
    parser.add_argument("--batch_size",                type=int,             default=2048,            help="Total train batch size (w. parallel, distributed & accumulation).")
    parser.add_argument("--batch_gpu",                 type=int,             default=16,              help="Batch size per device for training.")
    parser.add_argument("--batch_vae",                 type=int,             default=16,              help="VAE Batch size per device for training.")
    parser.add_argument("--sp_size",                   type=int,             default=1,               help="Sequence parallel size.")
    parser.add_argument("--eval_batch_size",           type=int,             default=16,              help="Number of images for evaluation.")
    parser.add_argument("--num_epochs",                type=int,             default=100,             help="Total number of training epochs.")
    parser.add_argument("--ref_lr",                    type=float,           default=1e-4,            help="ref_lr of EDM2 learning rate scheduler.")
    parser.add_argument("--ref_batches",               type=int,             default=0,               help="ref_batches of EDM2 learning rate scheduler.")
    parser.add_argument("--rampup_Msample",            type=float,           default=0,               help="rampup_Msample of EDM2 learning rate scheduler.")
    parser.add_argument("--adam_beta1",                type=float,           default=0.95,            help="Adam optimizer beta1.")
    parser.add_argument("--adam_beta2",                type=float,           default=0.999,           help="Adam optimizer beta2.")
    parser.add_argument("--adam_weight_decay",         type=float,           default=1e-6,            help="Adam optimizer weight decay.")
    parser.add_argument("--adam_epsilon",              type=float,           default=1e-08,           help="Adam optimizer epsilon.")
    parser.add_argument("--ema_decay",                 type=str,             default="",              help="EMA decay list, e.g., '0.99,0.999'.")
    parser.add_argument("--offload_ema",               action="store_true",  default=False,           help="Whether to move EMA to cpu.")
    parser.add_argument("--logger",                    type=str,             default="tensorboard",   help="Logger type: 'tensorboard' or 'wandb'.", choices=["tensorboard", "wandb"], )
    parser.add_argument("--logging_dir",               type=str,             default="logs",          help="TensorBoard log directory.")
    parser.add_argument("--local_rank",                type=int,             default=-1,              help="Local rank for distributed training.")
    parser.add_argument("--checkpointing_steps",       type=int,             default=500,             help="Save checkpoint every X updates.")
    parser.add_argument("--checkpoints_total_limit",   type=int,             default=None,            help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint",    type=str,             default=None,            help="Path to checkpoint to resume from or 'latest'.")
    parser.add_argument("--passes",                    type=str,             default=None,            help="Comma-separated DeepSpeed compile passes (e.g. prefetch,selective_gather,offload_adam_states,offload_adam_states_sync).")
    parser.add_argument("--backend",                   type=str,             default="inductor",      help="Backend to use for model compilation.")
    parser.add_argument("--compile",                   action="store_true",  default=False,           help="Use torch.compile on the model.")
    parser.add_argument("--grad_checkpoint",           action="store_true",  default=False,           help="Use gradient checkpointing.")
    parser.add_argument("--reset_lr",                  action="store_true",  default=False,           help="Whether to reset the learning rate.")
    
    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings('ignore', '`data_source` argument is not used and will be removed')
        super().__init__(dataset)
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(order)
            yield int(order[idx % self.dataset_size])
            idx += self.stride

#----------------------------------------------------------------------------

from collections import OrderedDict

@torch.no_grad()
def update_ema_from_state_dict(ema_model, state_dict, decay):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, ema_param in ema_params.items():
        src = state_dict.get(name, None)
        src = src.to(device=ema_param.device, dtype=torch.float32)
        ema_param.mul_(decay).add_(src, alpha=1 - decay)

#----------------------------------------------------------------------------

def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_size = world_size // args.sp_size
    gradient_accumulation_steps = args.batch_size // (args.batch_gpu * dp_size)
    args.batch_gpu_total = args.batch_gpu * gradient_accumulation_steps
    assert args.batch_size == args.batch_gpu * dp_size * gradient_accumulation_steps
    if args.dynamic_res:
        assert args.batch_gpu == 1, "Dynamic resolution only supports batch_gpu=1."
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=args.logger,
        project_config=accelerator_project_config
    )
    device = accelerator.device
    
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)
        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Setup distributed state.
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
    
    # Mixed precision.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        
    #------------------------------------------------------------------------------
    # Initialize the model.
    
    from training.models import WanTransformer3DModel
    from training.models.precond import (
        Precond,
        Wan2_1SpatialTiledEncoder3D,
        Wan2_1SpatialTiledConditionEncoder3D,
        Wan2_1SpatialTiledConditionEncoder3Dv2,
        MaskSpatialTiledEncoder3D,
    )
    from training.util import print_module_summary
    from safetensors import safe_open
    
    model_cls = WanTransformer3DModel

    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    # in_channels: noise + lq + msk
    in_channels = args.vae_channels + args.lq_proj_hidden_dim + args.temporal_compression_ratio
    transformer = model_cls.from_config(
        model_config,
        in_channels=in_channels,
        out_channels=args.vae_channels,
    )

    if args.model_path is not None:
        state_dict = {}
        files = []
        for fname in os.listdir(args.model_path):
            if fname.endswith('.safetensors'):
                files.append(os.path.join(args.model_path, fname))
        files = sorted(files)
        for shard_path in files:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in state_dict:
                        state_dict[key] = f.get_tensor(key)
        
        w_ckpt = state_dict['patch_embedding.weight']
        w_model = transformer.patch_embedding.weight
        new_w = torch.zeros_like(w_model)
        new_w[:, :w_ckpt.shape[1]] = w_ckpt
        state_dict['patch_embedding.weight'] = new_w

        b_ckpt = state_dict['patch_embedding.bias']
        b_model = transformer.patch_embedding.bias
        new_b = torch.zeros_like(b_model)
        new_b[:b_ckpt.shape[0]] = b_ckpt
        state_dict['patch_embedding.bias'] = new_b

        logger.info(transformer.load_state_dict(state_dict, strict=False))
            
    if args.attention_type is not None:
        transformer.set_attention_type(args.attention_type)
        logger.info(f"Set attention type to {args.attention_type}")
    
    if args.grad_checkpoint:
        transformer.enable_gradient_checkpointing()
    
    if args.lora_rank > 0:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v"],
        )
        transformer.requires_grad_(False)
        transformer.add_adapter(peft_config)
    else:
        transformer.requires_grad_(True)

    transformer.to(torch.float32)
    
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
    lq_encoder = vae_map[args.vae_type](args.vae_path, args.vae_batch_size, **tiled_kwargs)
    msk_encoder = MaskSpatialTiledEncoder3D(**tiled_kwargs)
    
    vae.init(device)
    lq_encoder.init(device)
    model = Precond(
        transformer=transformer,
        vae=vae,
        lq_encoder=lq_encoder,
        msk_encoder=msk_encoder,
        diffusion_forcing=args.diffusion_forcing,
        t_shift=args.t_shift,
    )

    if accelerator.is_main_process:
        print_module_summary(model, 2)
        
    #------------------------------------------------------------------------------
    # Get the dataset.
    
    from training.data.stsr_dataset import VideoDataset, TemporalDegradation
    
    # GT video data.   
    meta_data = datasets.load_from_disk(args.data)
    meta_data = meta_data['train'] if isinstance(meta_data, datasets.DatasetDict) else meta_data
    data_kwargs = dict(
        num_frames  = args.num_frames,
        pn          = args.pn, 
        dynamic_res = args.dynamic_res,
        no_bicubic  = args.no_bicubic,
        p_vflip     = args.p_vflip,
        p_hflip     = args.p_hflip,
        p_transpose = args.p_transpose,
    )
    train_data = VideoDataset(meta_data, **data_kwargs)
    total_train_samples = round(len(train_data) * args.num_epochs)
    logger.info(f"Created dataset with parameters: \n{json.dumps(data_kwargs, indent=4)}")
        
    # Temporal degradation.
    temporal_opt = dict(
        temporal_sf=args.temporal_sf,
    )
    temporal_degradation = TemporalDegradation(**temporal_opt)
    logger.info(f"Created temporal degradation with parameters: \n{json.dumps(temporal_opt, indent=4)}")
    
    #------------------------------------------------------------------------------
    # Get the optimizer and LR scheduler.

    from accelerate.utils import DummyOptim
    
    # Initialize the optimizer
    if (
        accelerator.state.deepspeed_plugin is not None 
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        logger.info("optimizer detected in deepspeed config, using DummyOptim.")
        optimizer_cls = DummyOptim
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        params=[p for p in model.parameters() if p.requires_grad],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = lambda cur_nsample: \
        learning_rate_schedule(cur_nsample, args.batch_size, args.ref_lr, args.ref_batches, args.rampup_Msample)
    
    #------------------------------------------------------------------------------
    # Setup training state.

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Potentially load in the weights and states from a previous save
    state = argparse.Namespace(cur_nsample=0, global_step=0)
    if args.start_step is not None:
        logger.info(f"Overriding start step to {args.start_step}")
        state.global_step = args.start_step
        state.cur_nsample = state.global_step * args.batch_size
    path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        logger.info(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
    else:
        logger.info(f"Resuming from checkpoint {path}")
        with open(os.path.join(args.output_dir, path, 'state.json'), 'r') as f:
            vars(state).update(json.load(f))
                        
    # Prepare EMA.
    if accelerator.is_main_process and args.ema_decay:
        ema_decays = [float(x) for x in args.ema_decay.split(',') if x.strip()]
        if len(ema_decays) > 0:
            resume_dir = os.path.join(args.output_dir, path) if path is not None else None
            model.prepare_ema(ema_decays, resume_dir, device='cpu' if args.offload_ema else device)
            logger.info(
                f"Prepared EMA with decays={ema_decays}, device={'cpu' if args.offload_ema else str(device)}, resume_dir={resume_dir}"
            )

    # Set SP state.
    if args.sp_size > 1:
        model.set_sp_state(sp_group, sp_world_size, sp_rank, sp_src_rank)

    # Data loader.
    data_sampler = InfiniteSampler(
        train_data, start_idx=state.cur_nsample, shuffle=True, seed=args.seed,
        rank=accelerator.process_index, num_replicas=accelerator.num_processes, 
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_gpu, sampler=data_sampler,
        num_workers=2, prefetch_factor=2, pin_memory=True
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, _ = accelerator.prepare(model, optimizer, train_loader)
    if path is not None:
        accelerator.load_state(os.path.join(args.output_dir, path))

    #------------------------------------------------------------------------------
    # Main training loop.
            
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total train samples (w. repeat) = {total_train_samples}")
    
    from training.data.stsr_dataset import upsample_temporal
    
    def _prepare_batch(batch):
        gt = batch['gt'].to(device) # TCHW
        lq = []
        msk = []
        for x in gt:
            x, m = temporal_degradation.apply(x)
            x = upsample_temporal(x, m, mode=args.temporal_upsample)
            lq.append(x)
            msk.append(m)
        lq = torch.stack(lq, dim=0)
        msk = torch.stack(msk, dim=0)
        return {
            'gt': rearrange(gt, 'b t c h w -> b c t h w').byte().cpu(),
            'lq': rearrange(lq, 'b t c h w -> b c t h w').byte().cpu(),
            'msk': msk.bool().cpu()
        }
        
    # Train!
    progress_bar = tqdm(
        total=total_train_samples, initial=state.cur_nsample, disable=not accelerator.is_main_process
    )
    train_iterator = iter(train_loader)
    batch_buffer = []
    torch.cuda.empty_cache()
    while True:
        if progress_bar.n >= progress_bar.total:
            logger.info("Training complete.")
            break
        
        #----------------------------------------------
        # Optimize.
        
        # Set learning rate.
        offset = args.start_step * args.batch_size if args.reset_lr and args.start_step else 0
        lr = lr_scheduler(state.cur_nsample - offset)
        for g in optimizer.param_groups:
            g["lr"] = lr
        
        # Accumulate gradients.
        model.train()
        local_loss = 0.0
        local_item = 0
        times = argparse.Namespace(data=0, broadcast=0, vae=0, fwd=0, bwd=0, ema=0)
        for _ in range(gradient_accumulation_steps):
            with accelerator.accumulate(model):
                # Get data.
                with torch.no_grad():
                    t0 = time()
                    if args.sp_size > 1:
                        if len(batch_buffer) == 0:
                            batch = _prepare_batch(next(train_iterator))
                            batch_list = [None] * args.sp_size
                            dist.all_gather_object(batch_list, batch, sp_group)
                            batch_buffer.extend(batch_list)
                        batch = batch_buffer.pop(0)
                    else:
                        batch = _prepare_batch(next(train_iterator))
                    gt = batch['gt'].to(device=device, dtype=weight_dtype).div_(127.5).sub_(1)
                    lq = batch['lq'].to(device=device, dtype=weight_dtype).div_(127.5).sub_(1)
                    msk = repeat(batch['msk'], 'b t -> b 1 t h w', h=lq.shape[-2], w=lq.shape[-1]).to(device=device, dtype=weight_dtype)
                    times.data = time() - t0
                    
                # Compute loss.
                t0 = time()
                loss = model(lq, msk, gt)
                times.fwd += time() - t0
                t0 = time()
                accelerator.backward(loss.mean())
                times.bwd += time() - t0
                local_loss += loss.detach().sum().item()
                local_item += loss.numel()
            
        # Run optimizer step and update EMA.
        optimizer.step()
        optimizer.zero_grad()
        if accelerator.is_main_process and args.ema_decay:
            t0 = time()
            state_dict = accelerator.get_state_dict(model)
            model.update_ema(state_dict)
            times.ema += time() - t0
            
        #----------------------------------------------
        # Log and save.

        # torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        # Logging.
        global_loss = accelerator.reduce(torch.tensor([local_loss], device=device)).item()
        global_item = accelerator.reduce(torch.tensor([local_item], device=device)).item()
        state.loss = global_loss / global_item
        state.lr = lr
        if accelerator.is_main_process:
            for k, v in vars(times).items():
                vars(state)[f"time/{k}"] = v
            with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(vars(state)) + "\n")
        progress_bar.set_postfix(vars(state))
        accelerator.log(vars(state), step=state.cur_nsample)
        
        # Update state.
        progress_bar.update(args.batch_size)
        state.global_step += 1
        state.cur_nsample += args.batch_size
        
        # Save checkpoint.
        if (
            (state.global_step % args.checkpointing_steps == 0)
            or (progress_bar.n >= progress_bar.total)
        ):
            torch.cuda.empty_cache()
            save_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                with open(os.path.join(save_path, 'state.json'), 'w') as f:
                    json.dump(vars(state), f, indent=2, ensure_ascii=False)
                accelerator.unwrap_model(model).save_ema(save_path)
            
    accelerator.end_training()
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)