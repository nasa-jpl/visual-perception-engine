import argparse
import json
import os
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from cuda import cuda
from tqdm import tqdm

import transforms
from model_management.model_cards import ModelCard, ModelHeadCard
from model_management.registry import ModelRegistry
from model_management.util import PRECISION_MAP_TORCH
from transforms.dinov2_preprocessing import DINOV2PreprocessingTorch
from transforms.resize_to_image import ResizeAndToCV2Image
from vp_engine.engine import Engine
from utils.naming_convention import *

# This assumes the script is run from the root of the repository
REPOSITORY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_images(input_shape: tuple[int, int]) -> List[np.ndarray]:
    """
    Loads and resizes images from a list of directories, mimicking the procedure
    in vp_engine/example_input.py.
    """
    EXAMPLE_IMG_DIR1 = os.path.join(REPOSITORY_DIR, "resources", "cheetah", "frames")
    EXAMPLE_IMG_DIR2 = os.path.join(REPOSITORY_DIR, "tests", "resources", "object_detection", "inputs")
    input_dirs = [EXAMPLE_IMG_DIR1, EXAMPLE_IMG_DIR2]

    images = []
    for input_dir in input_dirs:
        for img_file in sorted(os.listdir(input_dir)):
            image = cv2.imread(os.path.join(input_dir, img_file))
            if image is not None:
                resized = cv2.resize(image, (input_shape[1], input_shape[0]))
                images.append(resized)
    return images


def run_vp_engine_benchmark(engine: Engine, images: List[np.ndarray], num_heads: int) -> List[float]:
    """Runs the benchmark using the visual perception engine."""
    timings = []

    for i, image in enumerate(images):
        start_time = time.perf_counter()
        engine.input_image(image, i)

        processed_by_head = [False] * num_heads
        while not all(processed_by_head):
            for head_idx in range(num_heads):
                if not processed_by_head[head_idx]:
                    output = engine.get_head_output(head_idx)
                    if output is not None:
                        processed_by_head[head_idx] = True

        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    return timings


def run_baseline_benchmark(
    framework: str,
    images: List[np.ndarray],
    fm_config: dict,
    heads_config: List[dict],
    registry: ModelRegistry,
    input_shape: tuple,
) -> List[float]:
    """Runs the benchmark for the specified baseline framework."""
    timings = []

    # Load models directly from the config
    fm_name = fm_config["canonical_name"]
    model_heads = [registry.load_model(head["canonical_name"]) for head in heads_config]
    
    # Get model cards
    fm_card = registry.get_registered_models()[fm_name]
    
    # Load preprocessing function
    preprocessing = DINOV2PreprocessingTorch(
        fm_signature=fm_card.input_signature,
        fm_type=PRECISION_MAP_TORCH[fm_card.precision],
        canonical_height=input_shape[0],
        canonical_width=input_shape[1],
    )
    
    # Load postprocessing functions for each head
    postprocessing = []
    for head_config in heads_config:
        head_card = registry.get_registered_models()[head_config["canonical_name"]]
        postproc_name = head_config.get("postprocessing_function", "DefaultPostprocessing")
        postproc_class = getattr(transforms, postproc_name)
        
        postproc_instance = postproc_class(
            mh_signature=head_card.output_signature,
            mh_type=PRECISION_MAP_TORCH[head_card.precision],
            canonical_height=input_shape[0],
            canonical_width=input_shape[1],
        )
        postprocessing.append(postproc_instance)

    # Define inference logic based on framework
    if "fully_sequential" in framework:
        foundation_models = [registry.load_model(fm_name) for _ in heads_config]
        def inference(gpu_tensor):
            for i, (foundation_model, head, postproc) in enumerate(zip(foundation_models, model_heads, postprocessing)):
                preprocessed = preprocessing({PREPROCESSING_INPUT: gpu_tensor})
                fm_output = foundation_model.forward_annotated(preprocessed)
                head_output = head.forward_annotated(fm_output)
                post_output = postproc(head_output)
                # Move all keys in output dict to CPU
                for key in post_output:
                    _ = post_output[key].cpu()

    elif "head_sequential" in framework:
        foundation_model = registry.load_model(fm_name)
        def inference(gpu_tensor):
            preprocessed = preprocessing({PREPROCESSING_INPUT: gpu_tensor})
            fm_output = foundation_model.forward_annotated(preprocessed)
            for i, (head, postproc) in enumerate(zip(model_heads, postprocessing)):
                head_output = head.forward_annotated(fm_output)
                post_output = postproc(head_output)
                # Move all keys in output dict to CPU
                for key in post_output:
                    _ = post_output[key].cpu()
                    
    else: # This handles merged frameworks
        merged_model = registry.load_model(fm_name)
        postproc = postprocessing[0]
        def inference(gpu_tensor):
            for i, (head, postproc) in enumerate(zip(model_heads, postprocessing)):
                preprocessed = preprocessing({PREPROCESSING_INPUT: gpu_tensor})
                output = head.forward_annotated(preprocessed)
                postprocessed_output = postproc(output)
                for key in postprocessed_output:
                    _ = postprocessed_output[key].cpu()

    for image in tqdm(images, desc=f"Processing with {framework}"):
        start_time = time.perf_counter()
        
        input_tensor_cpu = torch.tensor(image)
        input_tensor_gpu = input_tensor_cpu.to('cuda')

        inference(input_tensor_gpu)
        
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    return timings


def main():
    parser = argparse.ArgumentParser(description="Visual Perception Engine Performance Benchmark")
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
        help="Number of identical model heads to use.",
    )
    parser.add_argument(
        "--obj_det_heads",
        action='store_true',
        help="For baselines, use ObjectDetectionHead_fp16_torch instead of the default head.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.jsonl",
        help="Path to the output JSONL file for results.",
    )
    parser.add_argument(
        "--registry_path",
        type=str,
        default="model_registry/registry.jsonl",
        help="Path to the model registry file.",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.obj_det_heads and "merged" in args.framework:
        parser.error("--obj_det_heads cannot be used with 'merged' frameworks.")

    config_path = f"configs/benchmark_{args.num_heads}.json" if not args.obj_det_heads else f"configs/benchmark_obj_{args.num_heads}.json"
    
    # Load config file
    config_path = os.path.join(REPOSITORY_DIR, config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    input_shape = config['canonical_image_shape_hwc'][:2]

    model_type = "tensorrt" if "tensorrt" in args.framework else "torch"
    
    # Prepare model configurations for baselines
    num_heads = args.num_heads
    if "merged" in args.framework:
        postproc_name = "ResizeAndToCV2Image"
        fm_config = {"canonical_name": f"DepthAnythingV2_vits_{model_type}_fp16", "postprocessing_function": postproc_name}
        heads_config = [fm_config]* num_heads
    else:
        fm_config = {"canonical_name": f"DinoFoundationModel_fp16_{model_type}__encoder_size_vits__ignore_xformers_True"}
        if args.obj_det_heads:
            head_name = "ObjectDetectionHead_fp16_torch__encoder_size_vits"
            postproc_name = "DefaultPostprocessing"
        else:
            head_name = f"DepthAnythingV2Head_fp16_{model_type}__encoder_size_vits"
            postproc_name = "ResizeAndToCV2Image"
        heads_config = [{"canonical_name": head_name, "postprocessing_function": postproc_name}] * num_heads
        
    # Load images
    images = load_images(input_shape)

    # Run benchmark
    timings = []
    if args.framework == "vp_engine":
        # For the engine, the behavior is dictated by its config file.
        # The script flags control the baselines for comparison.
        engine_num_heads = len(config.get('model_heads', []))
        engine = Engine(config_path, os.path.join(REPOSITORY_DIR, args.registry_path))
        
        print("\n--- Configuration ---")
        print(f"Foundation model: {engine.config.foundation_model['canonical_name']}")
        for n, head_config in enumerate(engine.config.model_heads):
            print(f"Model head {n}: {head_config['canonical_name']}")
            
        try:
            engine.build()
            engine.start_inference()
            engine.test() # to warm up the engine
            timings = run_vp_engine_benchmark(engine, images, engine_num_heads)
        finally:
            engine.stop()
    else:
        
        print("\n--- Configuration ---")
        print(f"Foundation model: {fm_config['canonical_name']}")
        for n, head_config in enumerate(heads_config):
            print(f"Model head {n}: {head_config['canonical_name']}")
            
        registry = ModelRegistry(os.path.join(REPOSITORY_DIR, args.registry_path))
        timings = run_baseline_benchmark(args.framework, images, fm_config, heads_config, registry, input_shape)

    if not timings:
        print("No timings were recorded.")
        return

    # Calculate statistics
    mean_time = np.mean(timings)
    median_time = np.median(timings)
    p25_time = np.percentile(timings, 25)
    p75_time = np.percentile(timings, 75)

    results = {
        **vars(args),
        "mean_time": mean_time,
        "median_time": median_time,
        "25_percentile_time": p25_time,
        "75_percentile_time": p75_time,
        "fps": 1 / mean_time if mean_time > 0 else 0,
    }
    
    print("\n--- Benchmark Results ---")
    print(f"Framework: {args.framework}")
    print(f"Number of Heads: {num_heads}")
    print(f"Mean Time: {results['mean_time']:.4f} s")
    print(f"Median Time: {results['median_time']:.4f} s")
    print(f"25th Percentile: {results['25_percentile_time']:.4f} s")
    print(f"75th Percentile: {results['75_percentile_time']:.4f} s")
    print(f"Frames Per Second (FPS): {results['fps']:.2f}")

    # Append results to the output file
    output_path = os.path.join(REPOSITORY_DIR, args.output_file)
    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")
    
    print(f"\nResults appended to {output_path}")


if __name__ == "__main__":
    main()