#!/bin/bash

cd "$ROS_WORKSPACE"/nn_engine

for precision in  "fp16" "fp32"
do
    for encoder_size in "vits" "vitb" "vitl"
    do
        for framework in "xformers" "torch" "tensorrt"
        do
            for time in 30 0
            do
                python3 -m src.benchmarking.sequential_benchmark --model "depth_anything_v2_${encoder_size}_${precision}_${framework}" --eval_time $time
            done
        done
    done
done