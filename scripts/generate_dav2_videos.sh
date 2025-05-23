#! /bin/bash

output_dir="$ROS_WORKSPACE"/nn_engine/outputs/videos/"

vits_fp16="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vits_fp16_tensorrt/all_images/2024-11-25T14-27-19/output"
vits_fp32="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vits_fp32_tensorrt/all_images/2024-11-25T14-51-11/output"

vitb_fp16="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vitb_fp16_tensorrt/all_images/2024-11-25T14-35-26/output"
vitb_fp32="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vitb_fp32_tensorrt/all_images/2024-11-25T14-58-48/output"

vitl_fp16="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vitl_fp16_tensorrt/all_images/2024-11-25T14-44-12/output"
vitl_fp32="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vitl_fp32_tensorrt/all_images/2024-11-25T15-11-25/output"

vitl_fp32_torch="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vitl_fp32_torch/all_images/2024-11-25T15-05-31/output"
vits_fp16_torch="$ROS_WORKSPACE"/nn_engine/runs/cheetah_benchmark/depth_anything_v2_vits_fp16_torch/all_images/2024-11-25T14-22-41/output"

# Generate videos
cd "$ROS_WORKSPACE"/nn_engine/src/benchmarking/visualization

# compare different sizes
python3 generate_comparatory_video.py --directories $vits_fp16 $vitb_fp16 $vitl_fp16 --descriptions vits_fp16 vitb_fp16 vitl_fp16 --output_file "${output_dir}size_comparison_fp16.mp4"
python3 generate_comparatory_video.py --directories $vits_fp32 $vitb_fp32 $vitl_fp32 --descriptions vits_fp32 vitb_fp32 vitl_fp32 --output_file "${output_dir}size_comparison_fp32.mp4"

# compare different quantizations
python3 generate_comparatory_video.py --directories $vits_fp16 $vits_fp32 --descriptions vits_fp16 vits_fp32 --output_file "${output_dir}quantization_comparison_vits.mp4"
python3 generate_comparatory_video.py --directories $vitb_fp16 $vitb_fp32 --descriptions vitb_fp16 vitb_fp32 --output_file "${output_dir}quantization_comparison_vitb.mp4"
python3 generate_comparatory_video.py --directories $vitl_fp16 $vitl_fp32 --descriptions vitl_fp16 vitl_fp32 --output_file "${output_dir}quantization_comparison_vitl.mp4"

# compare best with worst
python3 generate_comparatory_video.py --directories $vits_fp16 $vitl_fp32 --descriptions vits_fp16 vitl_fp32 --output_file "${output_dir}vits_fp16_vs_vitl_fp32.mp4"

# compare torch with tensorrt
python3 generate_comparatory_video.py --directories $vitl_fp32 $vitl_fp32_torch --descriptions vitl_fp32 vitl_fp32_torch --output_file "${output_dir}vitl_fp32_vs_vitl_fp32_torch.mp4"
python3 generate_comparatory_video.py --directories $vits_fp16 $vits_fp16_torch --descriptions vits_fp16 vits_fp16_torch --output_file "${output_dir}vits_fp16_vs_vits_fp16_torch.mp4"