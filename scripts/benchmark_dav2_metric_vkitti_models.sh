#!/bin/bash

cd "$ROS_WORKSPACE"/nn_engine

for precision in  "fp16" "fp32"
do
    for encoder_size in "vits" "vitb" "vitl"
    do
        for framework in "tensorrt" "xformers" "torch" 
        do
            python3 -m src.benchmarking.sequential_benchmark --model "depth_anything_v2_metric_vkitti_${encoder_size}_${precision}_${framework}" --benchmark kitti
        done
    done
done