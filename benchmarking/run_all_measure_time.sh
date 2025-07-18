#!/usr/bin/env bash

### IMPORTANT: this needs to be run within container for correct time estimates

### IMPORTANT: run the following outside container first for maximum power
# # make sure that jetson is set to maximum power mode
# sudo nvpmodel -m 0

# # make sure that fan is set to maximum speed
# sudo jetson_clocks --fan
###

# make sure CUDA MPS is on
export CUDA_VISIBLE_DEVICES=0 # Select GPU 0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps/log

# make sure that the directories exist and have correct permissions
sudo mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
sudo chown $USER $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

# Start the daemon
nvidia-cuda-mps-control -d

# Change dir to the src/
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../src

export OUTPUT_FILE=results_speed.jsonl

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
      echo "Running: python -m vp_engine.performance_benchmark --framework ${framework} --num_heads ${num_heads} --output_file ${SCRIPT_DIR}/${OUTPUT_FILE}" 
      python -m vp_engine.performance_benchmark --framework "${framework}" --num_heads "${num_heads}" --output_file $SCRIPT_DIR/$OUTPUT_FILE
    else
      # Non‑merged models: run twice, with and without the flag
      for obj_det in 0 1; do
        cmd="python -m vp_engine.performance_benchmark --framework ${framework} --num_heads ${num_heads} --output_file ${SCRIPT_DIR}/${OUTPUT_FILE}"
        [[ ${obj_det} -eq 1 ]] && cmd+=" --obj_det_heads"
        echo "Running: ${cmd}"
        eval "${cmd}"
      done
    fi

  done
done
