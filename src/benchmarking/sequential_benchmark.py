import argparse
import os
import time
import datetime
import numpy as np
import torch
from tqdm import tqdm
from typing import Callable, Optional

import cv2

from src import transforms
from src.transforms.abstract_transform import AbstractTransform
from src.model_management.registry import ModelRegistry
from src.model_management.model_cards import ModelCard
from src.model_management.util import PRECISION_MAP_TORCH
from src.transforms import ResizeAndToCV2Image
from .postprocessing import apply_color_map_and_save, compute_relative_error
from .helpers.resource_monitor import ResourceMonitoring
from src.nn_engine.naming_convention import *

# Useful paths
MODEL_REGISTRY = "model_registry/registry.jsonl"
OUTPUT_DIR = "nn_engine/runs"

# define all benchmarks
BENCHMARKS = {
    "cheetah": {
        "name": "cheetah_benchmark",
        "input_dir": "resources/cheetah/frames",
        "ground_truth_dir": None,
        "postprocessing_fn": apply_color_map_and_save,
    },
    "kitti": {
        "name": "kitti_benchmark",
        "input_dir": "resources/kitti_dataset/frames",
        "ground_truth_dir": "resources/kitti_dataset/ground_truth",
        "postprocessing_fn": compute_relative_error,
    },
}


def inference(
    model: torch.nn.Module,
    input: np.ndarray,
    preprocessing_func: Callable,
    postprocessing_fn: Callable,
    precision: torch.dtype,
) -> torch.Tensor:
    """Run inference on a single image."""
    eval_start_time = time.perf_counter()
    input_img = torch.tensor(input)
    img_cuda = input_img.to(device=torch.device("cuda"), dtype=precision)
    preprocessed = preprocessing_func({PREPROCESSING_INPUT: img_cuda})
    inference_start_time = time.perf_counter()
    out = model(preprocessed)
    inference_end_time = time.perf_counter()
    out = postprocessing_fn(
        out
    )  # NOTE: ensure that this is executed on the same stream as model(img_cuda) otherwise output might not be ready
    out = out[POSTPROCESSING_OUTPUT].cpu()
    eval_end_time = time.perf_counter()
    return out, eval_end_time - eval_start_time, inference_end_time - inference_start_time


def run(
    model,
    representative_model_card: ModelCard,
    preprocessing_func: AbstractTransform,
    outdir: str,
    evalutation_time: Optional[float],
    benchmark: dict,
):
    """Run the benchmark on the given model card and benchmark configuration.
    count_preprocessing: if True, the preprocessing time will be added to the inference time. Otherwise, only the inference time will be measured."""
    # Load the model
    precision = PRECISION_MAP_TORCH[representative_model_card.precision]

    # Create the output directory if it doesn't exist
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    type_of_eval = f"eval_time_{evalutation_time}s" if bool(evalutation_time) else "all_images"
    output_dir = os.path.join(outdir, benchmark["name"], representative_model_card.name, type_of_eval, time_stamp)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    ## Load the images
    inputs = []
    outputs = []
    img_files = sorted(os.listdir(benchmark["input_dir"]))
    for img_path in tqdm(img_files, desc="Loading images"):
        loaded_img = cv2.imread(os.path.join(benchmark["input_dir"], img_path))
        orig_h, orig_w = loaded_img.shape[:2]
        inputs.append(loaded_img)

    preprocessing_func_init = preprocessing_func(output_precision=precision, target_width=518, target_height=518)
    postprocessing_fn = ResizeAndToCV2Image(target_height=orig_h, target_width=orig_w)

    with ResourceMonitoring(log_dir) as metrics:
        if not bool(evalutation_time):
            # Evaluate all images
            for _ in tqdm(range(len(inputs)), desc="Inference"):
                input_img = inputs.pop(0)
                out, total_time, inf_time = inference(
                    model, input_img, preprocessing_func_init, postprocessing_fn, precision
                )
                metrics["inference_times"].append(inf_time)
                metrics["total_processing_times"].append(total_time)
                outputs.append(out)
        else:
            # Evaluate for a fixed amount of time
            pbar = tqdm(total=0)
            start_time = time.perf_counter()
            counter = 0
            while time.perf_counter() - start_time < evalutation_time:
                input_img = inputs[counter % len(inputs)]
                out, total_time, inf_time = inference(
                    model, input_img, preprocessing_func_init, postprocessing_fn, precision
                )
                metrics["inference_times"].append(inf_time)
                metrics["total_processing_times"].append(total_time)
                outputs.append(out)
                counter += 1
                pbar.update(1)

    for metric in metrics:
        print("#" * 10, metric, "#" * 10)
        print(f"speed: {1 / np.mean(metrics[metric]):.2f}")
        print(f"time: {np.mean(metrics[metric]):.4f}")
        print(f"time (sd): {np.std(metrics[metric]):.4f}")

    post_fn = benchmark["postprocessing_fn"]
    if post_fn is not None:
        post_fn(outputs, orig_h, orig_w, output_dir, benchmark, img_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a model by running inference on a sequence of images")
    parser.add_argument("--benchmark", choices=BENCHMARKS.keys(), default="cheetah", help="The benchmark to run")
    parser.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Output directory for the benchmarking results")
    parser.add_argument("--model_registry", type=str, default=MODEL_REGISTRY, help="Path to the model registry")
    parser.add_argument(
        "--preprocessing_func", type=str, default="DINOV2PreprocessingTorch", help="The preprocessing function to use"
    )
    parser.add_argument("--model", type=str, help="The name of a model in the model registry that will be benchmarked")
    parser.add_argument(
        "--fm", type=str, help="The name of a foundation model in the model registry that will be benchmarked"
    )
    parser.add_argument(
        "--mh",
        type=str,
        help="The name of a model head in the model registry that will be benchmarked together with the foundation model",
    )
    parser.add_argument(
        "--eval_time",
        type=float,
        help="If set the model will be evaluated for approximately this amount of time. Otherwise, the model will be evaluated on all images and then stop",
    )

    args = parser.parse_args()
    assert not args.eval_time or args.eval_time <= 30, (
        "Evaluation time should be less or equal to 30 seconds, because tensorrt runs at ~ 120Hz"
    )
    assert args.model or (args.fm and args.mh), "Either a model or a foundation model must be specified"

    # Retrieve the correct model card
    registry = ModelRegistry(args.model_registry)
    preprocessing_func = getattr(transforms, args.preprocessing_func)

    if args.model:
        model_card = registry.get_registered_models()[args.model]
        model = ModelRegistry.load_model_from_card(model_card)

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.forward_annotated(x)

        model = Wrapper(model)
        run(model, model_card, preprocessing_func, args.outdir, args.eval_time, BENCHMARKS[args.benchmark])
    elif args.fm and args.mh:
        fm_card = registry.get_registered_models()[args.fm]
        mh_card = registry.get_registered_models()[args.mh]

        fm_model = ModelRegistry.load_model_from_card(fm_card)
        mh_model = ModelRegistry.load_model_from_card(mh_card)

        class JointModel(torch.nn.Module):
            def __init__(self, fm_model, mh_model):
                super().__init__()
                self.fm_model = fm_model
                self.mh_model = mh_model

            def forward(self, x):
                x = self.fm_model.forward_annotated(x)
                x = self.mh_model.forward_annotated(x)
                return x

        model = JointModel(fm_model, mh_model)

        run(model, mh_card, preprocessing_func, args.outdir, args.eval_time, BENCHMARKS[args.benchmark])
    else:
        raise ValueError("Either a model or a foundation model must be specified")
