#!/bin/bash
#------------------------------

distributed_args="
    --num_machines=$nnodes
    --num_processes=$((nnodes * nproc_per_node))
    --machine_rank=$node_rank
    --main_process_ip=$master_addr
    --main_process_port=$master_port
"

model_path="/path/to/Wan2.1-T2V-1.3B-Diffusers/transformer"
vae_path="/path/to/Wan2.1_VAE.pth"
data_path="data/lavib-hf"

output_dir="runs/train_dit"

#------------------------------

model_args="
    --model_config=configs/wan1_3B.json
    --model_path=$model_path
    --vae_channels=16
    --lq_proj_hidden_dim=16
    --attention_type=slide_chunk_all_block_2x1x1
    --temporal_upsample=nearest
    --diffusion_forcing=temporal
    --num_frames=60
"

vae_args="
    --vae_type=wan2_1
    --vae_path=$vae_path
    --vae_batch_size=16
    --tile_min_h=256
    --tile_min_w=256
    --tile_min_t=20
    --tile_stride_h=192
    --tile_stride_w=192
    --spatial_compression_ratio=8
    --temporal_compression_ratio=4
"

data_args="
    --data=$data_path
    --pn=0.25M
    --no_bicubic
    --temporal_sf=2-16
    --avg_frames=0,0
    --p_vflip=0.0
    --p_hflip=0.0
    --p_transpose=0.0
"

train_args="
    --batch_size=256
    --ref_lr=5e-5
    --rampup_Msample=0.5
    --t_shift=5
    --num_epochs=20
    --start_step=0
    --reset_lr
    --ema_decay=0.995,0.999
"

performance_args="
    --batch_gpu=8
    --batch_vae=32
    --sp_size=1
    --grad_checkpoint
    --checkpointing_steps=100
    --resume_from_checkpoint=latest
"

mkdir -p $output_dir
echo "output_dir: $output_dir"
accelerate launch --config_file="configs/zero1.yaml" $distributed_args \
    train_dit.py \
    --output_dir=$output_dir \
    $model_args \
    $vae_args \
    $data_args \
    $train_args \
    $performance_args \
    2>&1 | tee -a $output_dir/log.txt

#------------------------------
