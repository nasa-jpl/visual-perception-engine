import os
import numpy as np
import torch
from torch2trt import TRTModule

import model_architectures
from model_management.registry import ModelRegistry
from model_management.model_cards import ModelCard
from .postprocessing import *
from .sequential_benchmark import MODEL_REGISTRY, BENCHMARKS, PRECISION_MAP
from .helpers.transform import load_image

DEBUG_DIR = "runs/debuging"


# load trt model
def load_trt(model_card: ModelCard) -> TRTModule:
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(model_card.path2weights))
    trt_model.eval()
    return trt_model


# load pytorch model
def load_pytorch(model_card: ModelCard) -> torch.nn.Module:
    model = getattr(model_architectures, model_card.model_class_name)(**model_card.init_arguments)
    model.load_state_dict(torch.load(model_card.path2weights))
    model = model.to(dtype=PRECISION_MAP[model_card.precision], device=torch.device("cuda"))
    model.eval()
    return model


def _compare_outputs(output1: np.ndarray, output2: np.ndarray) -> float:
    abs_diff = np.maximum(output1, output2) - np.minimum(output1, output2)
    return np.mean(abs_diff), np.max(abs_diff), np.sum(abs_diff)


def convert_to_np_unit8(depth: torch.Tensor) -> np.ndarray:
    depth = np.array(depth.cpu().detach()).squeeze()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth


def main(total_frames: int, model1: dict, model2: dict):
    # load model registry
    model_registry = ModelRegistry(MODEL_REGISTRY)

    # load benchmark
    benchmark = BENCHMARKS["cheetah"]

    # load model cards
    model_cards = [
        model_registry.get_registered_models()[
            f"depth_anything_v2_{model['encoder_size']}_{model['precision']}_{model['framework']}"
        ]
        for model in [model1, model2]
    ]

    # load trt models
    models = [
        load_trt(model_card) if model_card.framework == "tensorrt" else load_pytorch(model_card)
        for model_card in model_cards
    ]

    imgs = sorted(os.listdir(benchmark["input_dir"]))

    total_frames = min(total_frames, len(imgs))

    means = []
    maxs = []
    sums = []
    for idx, img in enumerate(tqdm(imgs[:total_frames], desc="Processing frames")):
        input_img, (orig_h, orig_w) = load_image(
            os.path.join(benchmark["input_dir"], img),
            model_cards[0].canonical_input_shape[0],
            model_cards[0].canonical_input_shape[1],
        )
        inputs = [
            torch.tensor(input_img.copy(), device=torch.device("cuda"), dtype=PRECISION_MAP[model_card.precision])
            for model_card in model_cards
        ]
        outputs = [convert_to_np_unit8(model(input)) for model, input in zip(models, inputs)]
        mean_diff, max_diff, sum_diff = _compare_outputs(outputs[0], outputs[1])
        means.append(mean_diff)
        maxs.append(max_diff)
        sums.append(sum_diff)
        # print(f"Frame {idx:04} -- Mean difference: {mean_diff}, Max difference: {max_diff}, Sum difference: {sum_diff}")

        # save difference of images to debug dir
        abs_diff = np.maximum(outputs[0], outputs[1]) - np.minimum(outputs[0], outputs[1])
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{idx:04}_diff.png"), abs_diff)

    print(
        f"Averaged over {total_frames} frames -- Mean difference: {round(np.mean(means), 3)}, Max difference: {round(np.mean(maxs), 3)}, Sum difference: {round(np.mean(sums), 3)}"
    )


if __name__ == "__main__":
    config = {
        "total_frames": 1000,
        "model1": {
            "framework": "torch",
            "precision": "fp16",
            "encoder_size": "vits",
        },
        "model2": {
            "framework": "torch",
            "precision": "fp32",
            "encoder_size": "vits",
        },
    }

    main(**config)

    print("Configuration: ", config)
