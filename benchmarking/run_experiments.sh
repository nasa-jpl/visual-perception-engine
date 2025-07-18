#!/usr/bin/env bash

frameworks=(
  vp_engine
  fully_sequential_torch
  fully_sequential_tensorrt
  head_sequential_torch
  head_sequential_tensorrt
  merged_torch
  merged_tensorrt
)

for framework in "${frameworks[@]}"; do
  for num_heads in {1..8}; do

    # Decide whether the obj‑det flag is even legal for this framework
    if [[ ${framework} == merged_* ]]; then
      # Merged models: run once, without the flag
      echo "Running: python run_with_gpu_monitoring.py --framework ${framework} --num_heads ${num_heads}"
      python run_with_gpu_monitoring.py --framework "${framework}" --num_heads "${num_heads}"
    else
      # Non‑merged models: run twice, with and without the flag
      for obj_det in 0 1; do
        cmd="python run_with_gpu_monitoring.py --framework ${framework} --num_heads ${num_heads}"
        [[ ${obj_det} -eq 1 ]] && cmd+=" --obj_det_heads"
        echo "Running: ${cmd}"
        eval "${cmd}"
      done
    fi

  done
done
