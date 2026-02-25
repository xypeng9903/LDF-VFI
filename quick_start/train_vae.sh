#!/bin/bash
#------------------------------

distributed_args="
    --num_machines=4
    --num_processes=32
    --machine_rank=$node_rank
    --main_process_ip=$master_addr
    --main_process_port=$master_port
"

vae_path="/path/to/Wan2.1_VAE.pth"
data_path="data/lavib-hf"

output_dir="runs/train_vae"

#------------------------------

vae_args="
    --vae_path=$vae_path
    --temporal_upsample=nearest
    --freeze_encoder
    --disc_start=5001
"

data_args="
    --data=$data_path
    --pn=0.06M
    --num_frames=17
    --temporal_sf=1-16
    --avg_frames=0,0
    --no_bicubic
    --p_vflip=0.0
    --p_hflip=0.0
    --p_transpose=0.0
"

train_args="
    --batch_size=256
    --ref_lr=5e-5
    --disc_lr=5e-5
    --rampup_Msample=0.1
    --num_epochs=100
    --ema_decay=0.999,0.9995
    --reset_lr
"

performance_args="
    --batch_gpu=2
    --checkpointing_steps=1000
    --resume_from_checkpoint=latest
"

mkdir -p $output_dir
echo "output_dir: $output_dir"
accelerate launch --config_file="configs/zero1.yaml" --mixed_precision='bf16' $distributed_args \
    train_vae.py \
    --output_dir=$output_dir \
    $vae_args \
    $data_args \
    $train_args \
    $performance_args \
    2>&1 | tee -a $output_dir/log.txt

#------------------------------
