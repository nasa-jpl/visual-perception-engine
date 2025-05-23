#!/bin/bash

cd "$ROS_WORKSPACE"/nn_engine

for precision in  "fp16" "fp32"
do
    for encoder_size in "vits" "vitb" "vitl"
    do
        for framework in "xformers" "torch" "tensorrt"
        do
            python3 -m src.model_management.export_dav2_model --precision $precision --encoder_size $encoder_size --framework $framework
            python3 -m src.model_management.export_dav2_metric_model --precision $precision --encoder_size $encoder_size --framework $framework
        done
    done
done