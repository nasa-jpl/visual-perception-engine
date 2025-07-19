import os
import shutil
from argparse import ArgumentParser
from queue import Empty, Full
from time import perf_counter, sleep
from typing import Literal

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from cuda import cuda

import model_architectures, transforms
from model_management.model_cards import ModelCard, ModelHeadCard
from model_management.registry import ModelRegistry
from model_management.util import PRECISION_MAP_TORCH
from vp_engine.log_analyzer import LogAnalyzer
from utils.logging_utils import MESSAGE, create_logger
from vp_engine.model_head import ModelHead
from vp_engine.threading_utils import QueueReceiverThread
from transforms import AbstractPostprocessing, AbstractPreprocessing
from vp_engine.engine import Engine

# NOTE This will not work from within an installed package
REPOSITORY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def measure_time(engine: Engine, max_wait_per_image_s: float = 60.0, example_input_max_images: int = 1000, input_dir: str = os.path.join(REPOSITORY_DIR, "resources", "cheetah", "frames")):    
    engine.process_names["input"] = engine.process_names["main"]
    example_images_data = []
    input_shape = engine.config.canonical_image_shape_hwc[:2]
    for img in sorted(os.listdir(input_dir)):
        image = cv2.imread(os.path.join(input_dir, img))
        resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        example_images_data.append(resized)
        
    num_total_images = len(example_images_data)
    
    timings = []
    num_heads = len(engine.model_heads)

    for i, image_data in enumerate(example_images_data):
        image_id = i # Use index as a unique image_id for this run

        current_image = image_data if isinstance(image_data, torch.Tensor) else torch.tensor(image_data)

        if current_image.dtype != torch.uint8:
            current_image = current_image.to(torch.uint8)

        processed_by_head = [False] * num_heads
        
        start_time = perf_counter()
        engine.input_image(current_image, image_id)
        
        while not all(processed_by_head) and (perf_counter() - start_time) < max_wait_per_image_s:
            for head_idx in range(num_heads):
                if not processed_by_head[head_idx]:
                    try:
                        retrieved_n, _ = engine.output_queues[head_idx].get_nowait() # We don't need the content
                        engine.logger.info(MESSAGE.OUTPUT_RECEIVED.format(n=retrieved_n, head_name=engine.model_heads[head_idx].name))
                        assert retrieved_n == image_id
                        processed_by_head[head_idx] = True
                    except Empty:
                        pass # Queue is empty for this head, continue polling

        end_time = perf_counter()

        duration = end_time - start_time
        timings.append(duration)

    if timings:
        mean_time = np.mean(timings)
        std_dev_time = np.std(timings)
        print("--- Time Measurement Results ---")
        print(f"Successfully processed and timed {len(timings)} images.")
        print(f"Mean processing speed per image: {1/mean_time:.4f} Hz")
        print(f"Mean processing time per image: {mean_time:.4f} seconds")
        print(f"Standard deviation of processing time: {std_dev_time:.4f} seconds")


def stress_test(engine: Engine, n_inputs: int, time_interval: float, timeout: float, save_outputs: bool):
    from vp_engine.example_input import ExampleInput

    example_input = ExampleInput(
        engine,
        time_interval=time_interval,
        max_inputs=n_inputs,
        input_shape=engine.config.canonical_image_shape_hwc[:2],
    )
    example_input.start()
    engine.input_queue.send_shareable_handles([example_input.pid])

    outputs = [[] for _ in range(len(engine.model_heads))]

    threads = []
    for i, output_queue in enumerate(engine.output_queues):
        thread = QueueReceiverThread(
            output_queue, outputs[i], n_inputs, engine.model_heads[i].name, engine.logger, timeout=timeout, store_output=save_outputs
        )
        threads.append(thread)

        thread.start()

    for thread in threads:
        thread.join()

    example_input.terminate()

    # Save the outputs to a file
    
    if save_outputs:
        engine.logger.info(MESSAGE.SAVING_OUTPUTS)
        for i, head_output in enumerate(outputs):
            head_dir = os.path.join(engine.config.output_dir, f"{engine.model_heads[i].name}")
            if os.path.exists(head_dir):
                shutil.rmtree(head_dir)
            os.makedirs(head_dir)

            for n, output in head_output:
                if not output:
                    continue
                try:
                    head_cls = getattr(model_architectures, engine.model_heads[i].model_card.model_class_name)
                    original_image = torch.tensor(example_input.input_images[n % len(example_input.input_images)])
                    output_img = head_cls.visualize_output(output, original_image)
                    cv2.imwrite(os.path.join(head_dir, f"{n:04d}.png"), output_img)
                except NotImplementedError:
                    break

        engine.logger.info(MESSAGE.OUTPUTS_SAVED)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.json", help="Path to the configuration file of the engine"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="model_registry/registry.jsonl",
        help="Path to the file containing the model registry",
    )
    parser.add_argument("--measure_time", action="store_true", help="Whether to measure how fast the engine is running instead of stress testing it")
    parser.add_argument("--n_inputs", type=int, default=100, help="Number of inputs to run the example with")
    parser.add_argument("--time_interval", type=float, default=1 / 30, help="Time interval between inputs")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for the example run")
    parser.add_argument("--save_outputs", action="store_true", help="Flag to save outputs after stress test. !!! Takes additional memory")
    args = parser.parse_args()

    args.config = os.path.join(REPOSITORY_DIR, args.config)
    args.registry = os.path.join(REPOSITORY_DIR, args.registry)

    if args.measure_time:
        engine = Engine(args.config, args.registry)
    else:
        engine = Engine(args.config, args.registry, reuse_input_queue_sockets = True)
        
    try:
        engine.build()

        # It is important to run few examples before actual serving for 2 reasons:
        # 1. To make sure that inference can be run
        # 2. First elements to be enqueued and dequeued take more time than the subsequent ones
        #    (it seems that on the first run new temp files are created)
        engine.start_inference()

        assert engine.test(args.timeout), "Engine test failed"

        if args.measure_time:
            measure_time(engine)
        else:
            stress_test(engine, n_inputs=args.n_inputs, time_interval=args.time_interval, timeout=args.timeout, save_outputs=args.save_outputs)

    finally:
        engine.stop()

    # Analyze the log
    LogAnalyzer(
        logging_config=engine.config.logging,
        input_proc=engine.process_names["input"],
        fm_proc=engine.process_names["foundation_model"],
        heads_proc=engine.process_names["model_heads"],
        main_proc=engine.process_names["main"],
    ).analyze_log()