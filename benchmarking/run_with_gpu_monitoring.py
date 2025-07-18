import os
import sys
import csv
import json
import subprocess
from argparse import ArgumentParser

from helpers.resource_monitor import GPUMonitoring

REPOSITORY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPOSITORY_DIR, "src")

def analyse_gpu_memory(csv_file: str):
    """
    Parse jtop CSV, find peak GPU‑memory usage minus baseline, and
    append a JSON‑line with the CLI arguments + result.

    Returns the delta (float, GiB) so you can log or print it.
    """
    baseline = peak = None

    with open(csv_file, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            mem = float(row["gpu_memory"])      # already in GiB
            if i == 0:
                baseline = peak = mem           # first line = baseline
            else:
                peak = max(peak, mem)

    delta = peak - baseline

    return delta

def main():
    parser = ArgumentParser(description="Visual Perception Engine Performance Benchmark")
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=[
            "vp_engine",
            "fully_sequential_torch",
            "fully_sequential_tensorrt",
            "head_sequential_torch",
            "head_sequential_tensorrt",
            "merged_torch",
            "merged_tensorrt",
        ],
        help="The framework to benchmark.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of identical model heads to use. Maximum 8.",
    )
    parser.add_argument(
        "--obj_det_heads",
        action='store_true',
        help="For baselines, use ObjectDetectionHead_fp16_torch instead of the default head.",
    )
    parser.add_argument(
        "--output_file_gpu_monitoring",
        type=str,
        default="results_gpu_memory.jsonl",
        help="Path to the output JSONL file for gpu memory results.",
    )
    parser.add_argument(
        "--output_file_speed",
        type=str,
        default="results_speed_outside_container.jsonl",
        help="Path to the output JSONL file for speed results.",
    )
    parser.add_argument(
        "--container", 
        default="6857100e15cd",
        type=str,
        help="Id of the container to attach to.",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.obj_det_heads and "merged" in args.framework:
        parser.error("--obj_det_heads cannot be used with 'merged' frameworks.")
        
    
    docker_cmd = ["docker", "exec", "--interactive"]
    cd_cmd = ["--workdir", SRC_DIR]
    exec_cmd = ["python", "-m", "vp_engine.performance_benchmark", "--framework", f"{args.framework}"]
    exec_cmd += ["--num_heads", f"{args.num_heads}"]
    exec_cmd += ["--obj_det_heads"] if args.obj_det_heads else []
    speed_res_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_file_speed)
    exec_cmd += ["--output_file", speed_res_filepath]
    
    cmd = docker_cmd + cd_cmd + [args.container] + exec_cmd
    
    tmp_file = "tmp.csv"
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)
        
    try:
        with GPUMonitoring(tmp_file):
            subprocess.run(cmd, check=True)

        # Analyse results:
        peak_memory_usage = analyse_gpu_memory(tmp_file)
        
        record = {
            "framework":     args.framework,
            "num_heads":     args.num_heads,
            "obj_det_heads": args.obj_det_heads,
            "peak_gpu_mem_GiB": round(peak_memory_usage, 4),
        }
        
        gpu_results_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_file_gpu_monitoring)
        with open(gpu_results_filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    
    
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

if __name__ == "__main__":
    main()