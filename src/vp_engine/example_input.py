import os
import time
import torch.multiprocessing as mp
from itertools import cycle

import cv2
import torch
from cuda import cuda

from utils.logging_utils import create_logger
from vp_engine.engine import Engine
from utils.naming_convention import *
from vp_engine.cuda_utils import checkCudaErrors

# NOTE This will not work from within an installed package
REPOSITORY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXAMPLE_IMG_DIR1 = os.path.join(REPOSITORY_DIR, "tests", "resources", "inputs")
EXAMPLE_IMG_DIR2 = os.path.join(REPOSITORY_DIR, "tests", "resources", "object_detection", "inputs")


class ExampleInput(mp.Process):
    def __init__(
        self,
        engine: Engine,
        input_dirs: list[str] = [EXAMPLE_IMG_DIR1, EXAMPLE_IMG_DIR2],
        time_interval: float = 1.0 / 30,
        max_inputs: int = None,
        input_shape: tuple[int, int] = (1080, 1920),  # height, width
        **kwargs,
    ):
        super().__init__(daemon=True, **kwargs)
        self.engine = engine
        self.engine.process_names["input"] = self.name

        self.input_dirs = input_dirs
        self.time_interval = time_interval
        self.max_inputs = max_inputs

        self.input_images = []
        for input_dir in input_dirs:
            for img in sorted(os.listdir(input_dir)):
                image = cv2.imread(os.path.join(input_dir, img))
                resized = cv2.resize(image, (input_shape[1], input_shape[0]))
                self.input_images.append(resized)

        self.generator = cycle(self.input_images)

    def run(self):
        """Populate input queue with images from input_dir. This will go on forever."""
        checkCudaErrors(cuda.cuInit(0))

        # receive shareable handles in the new process for the CUDAQueue and CUDATimeBuffer to work
        self.engine.input_queue.recv_shareable_handles()
        self.engine.logger = create_logger(**self.engine.config.logging)

        counter = 0
        for img in self.generator:
            if self.max_inputs is not None and counter >= self.max_inputs:
                break

            start_t = time.perf_counter()

            self.engine.input_image(img, counter)
            counter += 1

            sleep_for = max(self.time_interval*0.8, self.time_interval - (time.perf_counter() - start_t))
            time.sleep(sleep_for)
