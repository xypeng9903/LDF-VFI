import argparse
import logging
import os
import json
import warnings
from time import time, perf_counter
import re
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from einops import rearrange, repeat

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration


import lpips

# Monkey patch: sqrt(sum(x^2) + eps) to avoid NaNs in SqrtBackward
def _normalize_tensor_with_eps(in_feat, eps: float = 1e-8):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True) + eps)
    return in_feat / norm_factor
    
lpips.normalize_tensor = _normalize_tensor_with_eps


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

    # VAE arguments
    parser.add_argument("--vae_path",                  type=str,             default=None,            help=None)
    parser.add_argument("--vae_type",                  type=str,             default=None,            help=None)
    parser.add_argument("--disc_lr",                   type=float,           default=1e-4,            help=None)
    parser.add_argument("--disc_start",                type=int,             default=50001,           help=None)
    parser.add_argument("--perceptual_weight",         type=float,           default=1.0,             help=None)
    parser.add_argument("--disc_weight",               type=float,           default=0.5,             help=None)
    parser.add_argument("--kl_weight",                 type=float,           default=1e-6,            help=None)
    parser.add_argument("--freeze_encoder",            action="store_true",  default=False,           help=None)
    parser.add_argument("--freeze_decoder",            action="store_true",  default=False,           help=None)

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
    parser.add_argument("--adam_weight_decay",         type=float,           default=0,               help="Adam optimizer weight decay.")
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
    parser.add_argument("--time_log_every",            type=int,             default=1,               help="Log detailed timing every N steps (1 = every step).")
    
    args = parser.parse_args()
    return args


def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr


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


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_size = world_size
    gradient_accumulation_steps = args.batch_size // (args.batch_gpu * dp_size)
    assert args.batch_size == args.batch_gpu * dp_size * gradient_accumulation_steps

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(Path(args.output_dir) / 'args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Verbosity
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # --------------------------------------------------------------------------
    # Model
    
    from training.models.vae2_1_cond import WanVAECondition_
    from training.models.vae2_1_cond_v2 import WanVAECondition_ as WanVAECondition_v2
    from training.models.discriminator import NLayerDiscriminator

    # VAE.
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0
    )
    vae_map = {
        'wan2_1_cond': WanVAECondition_,
        'wan2_1_cond_v2': WanVAECondition_v2,
    }
    assert args.vae_type in vae_map.keys(), f"Execpted args.vae_type {vae_map.keys()}. Got {args.vae_type}."
    model = vae_map[args.vae_type](**cfg).to(torch.float32)
    if args.vae_path:
        logger.info(f'loading {args.vae_path}')
        model.load_state_dict(torch.load(args.vae_path, map_location=device), strict=False)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
        model.conv1.requires_grad_(False)
    if args.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.conv2.requires_grad_(False)

    # Discriminator
    discriminator = NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=True).to(torch.float32)

    #------------------------------------------------------------------------------
    # Get the dataset.
    
    from training.data.stsr_dataset import VideoDataset, TemporalDegradation, SpatialDegradation
    import yaml
    
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
    
    # Spatial degradation.
    spatial_degradation = None
    if args.spatial_opt is not None:
        with open(args.spatial_opt, 'r') as f:
            spatial_opt = yaml.safe_load(f)
        spatial_degradation = SpatialDegradation(spatial_opt, device)
        logger.info(f"Created spatial degradation with parameters: \n{json.dumps(spatial_opt, indent=4)}")
    
    # Temporal degradation.
    temporal_opt = dict(
        temporal_sf=args.temporal_sf,
    )
    temporal_degradation = TemporalDegradation(**temporal_opt)
    logger.info(f"Created temporal degradation with parameters: \n{json.dumps(temporal_opt, indent=4)}")

    # --------------------------------------------------------------------------
    # Optimizers.

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_d = torch.optim.Adam(
        params=[p for p in discriminator.parameters() if p.requires_grad],
        lr=args.disc_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # --------------------------------------------------------------------------
    # Prepare training state

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    state = argparse.Namespace(cur_nsample=0, global_step=0)
    if args.start_step is not None:
        logger.info(f"Overriding start step to {args.start_step}")
        state.global_step = args.start_step
        state.cur_nsample = state.global_step * args.batch_size

    # Resume
    path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
    else:
        logger.info(f"Resuming from checkpoint {path}")
        with open(os.path.join(args.output_dir, path, 'state.json'), 'r') as f:
            vars(state).update(json.load(f))

    # Dataloader
    data_sampler = InfiniteSampler(
        train_data,
        start_idx=state.cur_nsample * 2, # model + discriminator
        shuffle=True,
        seed=args.seed,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_gpu,
        sampler=data_sampler,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # EMA
    ema_models = []
    ema_decays = [float(x) for x in args.ema_decay.split(',') if x.strip() != ""]
    if len(ema_decays) > 0 and accelerator.is_main_process:
        for decay in ema_decays:
            ema = vae_map[args.vae_type](**cfg)            
            if path is not None:
                try:
                    ema_path = os.path.join(args.output_dir, path, f"ema-{decay}.pth")
                    ema.load_state_dict(torch.load(ema_path, map_location='cpu'))
                    logger.info(f"Loaded EMA decay={decay} from {ema_path}")
                except Exception as e:
                    logger.warning(f"Failed loading EMA {decay} from {ema_path}: {e}")
                    ema.load_state_dict(model.state_dict())
            else:
                ema.load_state_dict(model.state_dict())
            ema.to(dtype=torch.float32, device='cpu' if args.offload_ema else device).requires_grad_(False)
            ema_models.append(ema)
    
    # model = torch.compile(model)
    # discriminator = torch.compile(discriminator)
    model, discriminator, optimizer, optimizer_d = accelerator.prepare(
        model, discriminator, optimizer, optimizer_d
    )
    if path is not None:
        accelerator.load_state(os.path.join(args.output_dir, path))
    last_layer = accelerator.unwrap_model(model).decoder.head[-1].weight

    from training.data.stsr_dataset import upsample_temporal

    def _prepare_batch(batch):
        gt = batch['gt'].to(device) # TCHW
        lq = []
        msk = []
        for x in gt:
            if spatial_degradation is not None:
                kernel_kwargs = spatial_degradation.get_kernels()
                rand_kwargs = spatial_degradation.get_random()
                x = spatial_degradation.apply(dict(gt=x, **kernel_kwargs, **rand_kwargs))
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

    from training.vae_loss import KLLPIPSWithDiscriminator

    loss_fn = KLLPIPSWithDiscriminator(
        disc_start=args.disc_start,
        kl_weight=args.kl_weight,
        disc_weight=args.disc_weight,
        perceptual_weight=args.perceptual_weight,
    )
    loss_fn.to(device=device, dtype=torch.float32)

    class _PosteriorWrapper:
        def __init__(self, mu, log_var):
            self.mu = mu
            self.log_var = log_var

        def kl(self):
            return -0.5 * (1 + self.log_var - self.mu.pow(2) - self.log_var.exp())

    # --------------------------------------------------------------------------
    # Train loop

    logger.info("***** Running VAE training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_gpu}")
    logger.info(f"  Total train batch size (w. distributed & accumulation) = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total train samples (w. repeat) = {total_train_samples}")

    progress_bar = tqdm(total=total_train_samples, initial=state.cur_nsample, disable=not accelerator.is_main_process)
    train_iterator = iter(train_loader)
    torch.cuda.empty_cache()
    while True:
        if progress_bar.n >= progress_bar.total:
            logger.info("Training complete.")
            break
        
        # Per-step timing accumulators
        times = argparse.Namespace(
            # data
            data_fetch=0.0,      # wait next(train_iterator)
            data_prep=0.0,       # _prepare_batch CPU transforms/degradations
            h2d=0.0,             # host->device copies
            # generator update
            g_fwd=0.0,           # model forward
            g_loss=0.0,          # loss compute time (generator phase)
            g_bwd=0.0,           # backward time (generator phase)
            g_opt=0.0,           # optimizer.step time (G)
            # ema
            ema=0.0,             # EMA update time
            # discriminator update
            d_pregen=0.0,        # generating recon for D
            d_loss=0.0,          # loss compute time (discriminator phase)
            d_bwd=0.0,           # backward time (D)
            d_opt=0.0,           # optimizer.step time (D)
        )

        # --------------------------------------------------------------------------
        # 1. Generator update

        lr = learning_rate_schedule(state.cur_nsample, args.batch_size, args.ref_lr, args.ref_batches, args.rampup_Msample)
        for g in optimizer.param_groups:
            g["lr"] = lr
        local_tot_loss = 0.0
        local_rec_loss = 0.0
        local_p_loss = 0.0
        local_kl_loss = 0.0
        local_g_loss = 0.0
        model.train()
        discriminator.eval()

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    t0 = perf_counter()
                    raw = next(train_iterator)
                    times.data_fetch += perf_counter() - t0

                    t0 = perf_counter()
                    batch = _prepare_batch(raw)
                    times.data_prep += perf_counter() - t0

                    t0 = perf_counter()
                    gt = batch['gt'].to(device=device, dtype=weight_dtype, non_blocking=True).div_(127.5).sub_(1)
                    cond = batch['lq'].to(device=device, dtype=weight_dtype, non_blocking=True).div_(127.5).sub_(1)
                    msk = repeat(batch['msk'], 'b t -> b 1 t h w', h=cond.shape[-2], w=cond.shape[-1]).to(device=device, dtype=weight_dtype, non_blocking=True)
                    times.h2d += perf_counter() - t0
                
                # Forward (G)
                t0 = perf_counter()
                x_recon, mu, log_var = model(gt, cond, msk)
                times.g_fwd += perf_counter() - t0

                t0 = perf_counter()
                frames_gt = rearrange(gt.float(), 'b c t h w -> (b t) c h w')
                frames_rec = rearrange(x_recon.float(), 'b c t h w -> (b t) c h w')
                post = _PosteriorWrapper(mu, log_var)
                loss_g, log_g = loss_fn(
                    discriminator=discriminator,
                    inputs=frames_gt,
                    reconstructions=frames_rec,
                    posteriors=post,
                    optimizer_idx=0,
                    global_step=state.global_step,
                    last_layer=last_layer,
                    cond=None,
                    weights=None,
                )
                times.g_loss += perf_counter() - t0

                # Backward (G)
                t0 = perf_counter()
                accelerator.backward(loss_g)
                times.g_bwd += perf_counter() - t0

                local_tot_loss += loss_g.detach().float().item()
                local_rec_loss += log_g.get('rec_loss', torch.tensor(0.0, device=device)).detach().float().item()
                local_p_loss += log_g.get('p_loss', torch.tensor(0.0, device=device)).detach().float().item()
                local_kl_loss += log_g.get('kl_loss', torch.tensor(0.0, device=device)).detach().float().item()
                local_g_loss += log_g.get('g_loss', torch.tensor(0.0, device=device)).detach().float().item()
        
        # Optimize.
        t0 = perf_counter()
        optimizer.step()
        times.g_opt += perf_counter() - t0
        if len(ema_decays) > 0 and accelerator.is_main_process:
            t0 = time()
            state_dict = accelerator.get_state_dict(model)
            with torch.no_grad():
                for ema in ema_models:
                    for k, p in ema.state_dict().items():
                        src = state_dict.get(k, None)
                        if src is None:
                            continue
                        src = src.to(device=p.device, dtype=torch.float32)
                        decay = float(ema_decays[ema_models.index(ema)])
                        p.mul_(decay).add_(src, alpha=1 - decay)
            times.ema += time() - t0

        # --------------------------------------------------------------------------
        # 2. Discriminator update
        
        disc_lr = learning_rate_schedule(state.cur_nsample, args.batch_size, args.disc_lr, args.ref_batches, args.rampup_Msample)
        for g in optimizer_d.param_groups:
            g["lr"] = disc_lr
        local_d_loss = 0.0
        model.eval()
        discriminator.train()

        # Accumulate gradients.
        optimizer_d.zero_grad(set_to_none=True)
        for _ in range(gradient_accumulation_steps):
            with accelerator.accumulate(discriminator):
                with torch.no_grad():
                    t0 = perf_counter()
                    raw = next(train_iterator)
                    times.data_fetch += perf_counter() - t0

                    t0 = perf_counter()
                    batch = _prepare_batch(raw)
                    times.data_prep += perf_counter() - t0

                    t0 = perf_counter()
                    gt = batch['gt'].to(device=device, dtype=weight_dtype, non_blocking=True).div_(127.5).sub_(1)
                    cond = batch['lq'].to(device=device, dtype=weight_dtype, non_blocking=True).div_(127.5).sub_(1)
                    msk = repeat(batch['msk'], 'b t -> b 1 t h w', h=cond.shape[-2], w=cond.shape[-1]).to(device=device, dtype=weight_dtype, non_blocking=True)
                    times.h2d += perf_counter() - t0

                    # Pre-generate recon for D
                    t0 = perf_counter()
                    x_recon, mu, log_var = model(gt, cond, msk)
                    times.d_pregen += perf_counter() - t0

                t0 = perf_counter()
                frames_gt = rearrange(gt.float(), 'b c t h w -> (b t) c h w')
                frames_rec = rearrange(x_recon.float(), 'b c t h w -> (b t) c h w')
                post = _PosteriorWrapper(mu, log_var)
                loss_d, log_d = loss_fn(
                    discriminator=discriminator,
                    inputs=frames_gt.detach(),
                    reconstructions=frames_rec.detach(),
                    posteriors=post,
                    optimizer_idx=1,
                    global_step=state.global_step,
                    last_layer=last_layer,
                    cond=None,
                    weights=None,
                )
                times.d_loss += perf_counter() - t0
                
                # Backward (D)
                t0 = perf_counter()
                accelerator.backward(loss_d)
                times.d_bwd += perf_counter() - t0
                local_d_loss += loss_d.detach().float().item()

        # Optimize.
        t0 = perf_counter()
        optimizer_d.step()
        times.d_opt += perf_counter() - t0

        # --------------------------------------------------------------------------
        # Logging.
        
        global_tot = accelerator.reduce(torch.tensor([local_tot_loss], device=device)).item()
        global_rec = accelerator.reduce(torch.tensor([local_rec_loss], device=device)).item()
        global_p = accelerator.reduce(torch.tensor([local_p_loss], device=device)).item()
        global_kl = accelerator.reduce(torch.tensor([local_kl_loss], device=device)).item()
        global_g = accelerator.reduce(torch.tensor([local_g_loss], device=device)).item()
        global_d = accelerator.reduce(torch.tensor([local_d_loss], device=device)).item()
        global_cnt = accelerator.reduce(torch.tensor([gradient_accumulation_steps], device=device)).item()

        state.loss   = global_tot / global_cnt
        state.rec    = global_rec / global_cnt
        state.lpips  = global_p / global_cnt
        state.kl     = global_kl / global_cnt
        state.g_loss = global_g / global_cnt
        state.d_loss = global_d / global_cnt
        state.lr = lr
        state.disc_lr = disc_lr
        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(vars(state)) + "\n")
        progress_bar.set_postfix(vars(state))
        accelerator.log(vars(state), step=state.cur_nsample)

        # Timing logs (main process only)
        if accelerator.is_main_process and (state.global_step % args.time_log_every == 0):
            time_metrics = {
                'step': state.global_step,
                'cur_nsample': state.cur_nsample,
                # data
                'time/data_fetch': times.data_fetch,
                'time/data_prep': times.data_prep,
                'time/h2d': times.h2d,
                # generator
                'time/g_fwd': times.g_fwd,
                'time/g_loss': times.g_loss,
                'time/g_bwd': times.g_bwd,
                'time/g_opt': times.g_opt,
                # ema
                'time/ema': times.ema,
                # discriminator
                'time/d_pregen': times.d_pregen,
                'time/d_loss': times.d_loss,
                'time/d_bwd': times.d_bwd,
                'time/d_opt': times.d_opt,
            }
            # Derived totals
            time_metrics['time/step_total'] = (
                times.data_fetch + times.data_prep + times.h2d +
                times.g_fwd + times.g_loss + times.g_bwd + times.g_opt +
                times.ema +
                times.d_pregen + times.d_loss + times.d_bwd + times.d_opt
            )
            # Write separate timing log
            with open(os.path.join(args.output_dir, "time_log.jsonl"), "a") as f:
                f.write(json.dumps(time_metrics) + "\n")
            # Also push to tracker
            accelerator.log(time_metrics, step=state.cur_nsample)

        # Update state.
        progress_bar.update(args.batch_size)
        state.global_step += 1
        state.cur_nsample += args.batch_size

        # Save checkpoint.
        if (state.global_step % args.checkpointing_steps == 0) or (progress_bar.n >= progress_bar.total):
            torch.cuda.empty_cache()
            save_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                with open(os.path.join(save_path, 'state.json'), 'w') as f:
                    json.dump(vars(state), f, indent=2, ensure_ascii=False)
                if len(ema_decays) > 0:
                    for ema, decay in zip(ema_models, ema_decays):
                        torch.save(ema.state_dict(), os.path.join(save_path, f"ema-{decay}.pth"))

        # --------------------------------------------------------------------------

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    # with torch.autograd.set_detect_anomaly(True):
    main(args)
