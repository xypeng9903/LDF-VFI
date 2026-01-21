#!/bin/bash
#------------------------------

distributed_args="
    --num_machines=1
    --num_processes=1
"
model_path="/path/to/LDF-VFI/transformer"
vae_path="/path/to/LDF-VFI/Wan2.1_VAE_cond_v2.pth"
data_path="assets/demo.mp4"
temporal_sf=8
output_dir="runs/generate"

#------------------------------

model_args="
    --model_path=$model_path
    --attention_type=slide_chunk_all_block_2x1x1
    --temporal_upsample=nearest
    --num_frames=60
"

vae_args="
    --vae_type=wan2_1_cond_v2
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

generate_args="
    --sampling_steps=16
    --t_shift=8
"

task_args="
    --data=$data_path
    --temporal_sf=$temporal_sf
    --output_dir=$output_dir
    --fps=30
"

performance_args="
    --sp_size=1
"

mkdir -p $output_dir
echo "output_dir: $output_dir"

accelerate launch $distributed_args generate.py \
    $model_args \
    $vae_args \
    $generate_args \
    $task_args \
    $performance_args \
    2>&1 | tee -a $output_dir/log.txt

#------------------------------