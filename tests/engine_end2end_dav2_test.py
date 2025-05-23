import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.nn_engine.engine import Engine
from utils import WorkspaceDirectory, image_directories_equal

RESOURCES_PATH = "tests/resources"
MODEL_REGISTRY = "$ROS_WORKSPACE"/nn_engine/model_registry/registry.jsonl"
CONGIG_PATH = f"{RESOURCES_PATH}/configs/dav2_depth_end2end_test.json"
INPUT_PATH = f"{RESOURCES_PATH}/inputs"
OUTPUT_PATH = f"{RESOURCES_PATH}/outputs/depth/DepthAnythingV2_vits_tensorrt_fp16"

WORKSPACE = "tests/temporary" # must match one in the config file


@pytest.mark.timeout(30)
def test_engine_end2end_dav2():
    torch.multiprocessing.set_start_method('spawn', force=True)
    with WorkspaceDirectory(WORKSPACE, ["logs", "outputs"]):
        engine = Engine(CONGIG_PATH, MODEL_REGISTRY)
        engine.build()
        engine.start_inference()
        assert engine.test()
        # there are 5 examples in the input directory
        # for some reason only in the test environment the time for the foundation model 
        # to receive first input is ~1s as opposed to ~0.003s in the subsequent ones.
        engine.run_example(5, 0.5)
        engine.stop_engine()

        assert image_directories_equal(OUTPUT_PATH, "tests/temporary/outputs/DepthAnythingV2Head")

if __name__ == "__main__":
    test_engine_end2end_dav2()