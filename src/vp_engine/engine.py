import os
import signal
from queue import Empty, Full
from time import perf_counter, sleep
from typing import Literal

import numpy as np
import torch
import torch.multiprocessing as mp
from cuda import cuda

import transforms, model_architectures
from model_management.model_cards import ModelCard, ModelHeadCard
from model_management.registry import ModelRegistry
from model_management.util import PRECISION_MAP_TORCH
from vp_engine.config import Config
from vp_engine.cuda_buffer import CUDATimeBuffer
from vp_engine.cuda_queue import CUDAQueue
from vp_engine.cuda_utils import checkCudaErrors
from vp_engine.dynamic_shape_queue import DynamicShapeQueue
from vp_engine.foundation_model import FoundationModel
from utils.logging_utils import MESSAGE, create_logger
from vp_engine.model_head import ModelHead
from utils.naming_convention import *
from utils.shape_utils import is_io_compatible
from transforms import AbstractPostprocessing, AbstractPreprocessing

# NOTE This will not work from within an installed package
REPOSITORY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Engine:
    """Exposed methods are:
    - build: Builds the engine with the given configuration
    - start_inference: Starts the inference procedure
    - stop: Stops the engine
    - test: Tests the engine
    - change_model_rate: Changes the inference rate of the given model
    - input_image: Inputs an image to the engine
    - get_head_output: Gets the output of the given model head
    - get_foundation_model_params: Gets the parameters of the foundation model
    - get_model_heads_params: Gets the parameters of the model heads

    all of them should be called from the same process Engine was created in.
    """

    def __init__(self, 
                 config_file: str = os.path.join(REPOSITORY_DIR, "configs", "default.json"), 
                 registry: ModelRegistry | str = os.path.join(REPOSITORY_DIR, "model_registry", "registry.jsonl"), 
                 embedded_in_ros2: bool = False,
                 reuse_input_queue_sockets: bool = False) -> None:
        self.config = Config(config_file)
        self.registry = registry if isinstance(registry, ModelRegistry) else ModelRegistry(registry)
        self.embedded_in_ros2 = embedded_in_ros2
        self.foundation_model: FoundationModel = None
        self.model_heads: list[ModelHead] = []
        self.logger = create_logger(**self.config.logging)
        self.process_names = {
            "main": "MainProcess",
            "input": None,
            "foundation_model": None,
            "model_heads": [],
        }  # useful for log analysis later on

        self.input_queue = None
        self.middle_queue = None
        self.output_queues = []

        self.main_process_id = os.getpid()
        self.stopping = False
        self.reuse_input_queue_sockets = reuse_input_queue_sockets

    @property
    def _is_main_process(self):
        return os.getpid() == self.main_process_id

    def _validate_pipeline(
        self,
        input_signature: dict[str, tuple[int]],
        preprocessing_f: AbstractPreprocessing,
        fm_model_card: ModelCard,
        head_model_cards: list[ModelHeadCard],
        postprocessing_fs: list[AbstractPostprocessing],
    ) -> None:
        # Validate the input shapes of the input data and prepreprocessing function
        if not is_io_compatible(input_signature, preprocessing_f.input_signature):
            raise ValueError(
                "The input shapes of the preprocessing function and the input shapes of the foundation model are not compatible."
            )

        # Validate the output shapes of the preprocessing function and the input shapes of the foundation model
        if not is_io_compatible(preprocessing_f.output_signature, fm_model_card.input_signature):
            raise ValueError(
                "The output shapes of the preprocessing function and the input shapes of the foundation model are not compatible."
            )

        for model_head_card, postprocessing_f in zip(head_model_cards, postprocessing_fs):
            # Validate the output shapes of the foundation model and the input shapes of the model heads
            if not is_io_compatible(fm_model_card.output_signature, model_head_card.input_signature):
                raise ValueError(
                    f"The output shapes of the foundation model and the input shapes of the model head {model_head_card.name} are not compatible."
                )

            # Check if the model head was trained on the foundation model
            if model_head_card.trained_on != fm_model_card.name:
                self.logger.warning(
                    f"The model head {model_head_card.name} was not trained on the foundation model {fm_model_card.name}. Make sure you know what you are doing."
                )

            # Validate the output shapes of the model heads and the input shapes of the postprocessing functions
            if not is_io_compatible(model_head_card.output_signature, postprocessing_f.input_signature):
                raise ValueError(
                    f"The output shapes of the model head {model_head_card.name} and the input shapes of the postprocessing function are not compatible."
                )

        return True

    def __getstate__(self):
        state = self.__dict__.copy()
        # remove the references to other processes when pickling
        state["foundation_model"] = state["foundation_model"].name
        state["model_heads"] = [head.name for head in state["model_heads"]]
        state["output_queues"] = None
        state["middle_queue"] = None
        return state

    def _load_model_cards(self):
        """Load the model cards of the foundation model and the model heads."""
        available_model_cards = self.registry.get_registered_models()

        fm_model_card = available_model_cards[self.config.foundation_model["canonical_name"]]

        model_heads = []
        for head in self.config.model_heads:
            model_heads.append(available_model_cards[head["canonical_name"]])

        return fm_model_card, model_heads

    def _load_auxilary_functions(
        self, fm_model_card: ModelCard, mh_model_cards: list[ModelHeadCard]
    ) -> tuple[AbstractPreprocessing, list[AbstractPostprocessing]]:
        # Get preprocessing function for the foundation model
        preprocessing_func_name = self.config.foundation_model.get("preprocessing_function", "DefaultPreprocessing")
        preprocessing_func = getattr(transforms, preprocessing_func_name)
        preprocessing_func = preprocessing_func(
            fm_signature=fm_model_card.input_signature,
            fm_type=PRECISION_MAP_TORCH[fm_model_card.precision],
            canonical_height=self.config.canonical_image_shape_hwc[0],
            canonical_width=self.config.canonical_image_shape_hwc[1],
        )

        # Get postprocessing functions for the model heads
        postprocessing_funcs = []
        for i, model_head_card in enumerate(mh_model_cards):
            postprocessing_func_name = self.config.model_heads[i].get(
                "postprocessing_function", "DefaultPostprocessing"
            )
            postprocessing_func = getattr(transforms, postprocessing_func_name)
            postprocessing_func = postprocessing_func(
                mh_signature=model_head_card.output_signature,
                mh_type=PRECISION_MAP_TORCH[model_head_card.precision],
                canonical_height=self.config.canonical_image_shape_hwc[0],
                canonical_width=self.config.canonical_image_shape_hwc[1],
            )
            postprocessing_funcs.append(postprocessing_func)

        return preprocessing_func, postprocessing_funcs

    def build(self):
        if not self.embedded_in_ros2:
            # Ros2 does signal handling on its own hence to avoid overwriting it
            # we only set it up if not embedded in ros2
            # if it is embedded in ros2, the signal is already handled by ros2
            # and the user should just make sure to call stop afterwards
            def signal_handler(sig, frame):
                if not self.stopping:
                    self.stopping = True
                    self.stop()
                    raise KeyboardInterrupt

            signal.signal(signal.SIGINT, signal_handler)

        # setup multiprocessing and cuda if not already done
        ctx = mp.set_start_method("spawn")
        checkCudaErrors(cuda.cuInit(0))

        input_signature = {PREPROCESSING_INPUT: self.config.canonical_image_shape_hwc}
        fm_model_card, head_model_cards = self._load_model_cards()
        self.logger.info(
            MESSAGE.ENGINE_BUILD_START.format(fm_model_name=fm_model_card.name, num_heads=len(head_model_cards))
        )

        # Load preprocessing and postprocessing functions
        preprocessing_f, postprocessing_fs = self._load_auxilary_functions(fm_model_card, head_model_cards)

        # Validate the combination of the foundation model, the model heads, and the processing functions
        self._validate_pipeline(input_signature, preprocessing_f, fm_model_card, head_model_cards, postprocessing_fs)

        ### Create interprocess communication structures
        self.input_queue = CUDATimeBuffer(
            max_size=self.config.queue_sizes["input"],
            dtype=torch.uint8,
            input_device="cpu",
            output_device="cuda",
            data_signature=input_signature,
            add_batch_dim=False,
        )
        self.middle_queue = CUDATimeBuffer(
            max_size=self.config.queue_sizes["intermediate"],
            dtype=PRECISION_MAP_TORCH[fm_model_card.precision],
            input_device="cuda",
            output_device="cuda",
            data_signature=fm_model_card.output_signature,
            add_batch_dim=False,
        )
        self.output_queues = []
        for i in range(len(head_model_cards)):
            static_output_shape = all([None not in v for v in postprocessing_fs[i].output_signature.values()])
            if static_output_shape:
                queue = CUDAQueue(
                    max_size=self.config.queue_sizes["output"],
                    dtype=postprocessing_fs[i].output_type,
                    input_device="cuda",
                    output_device="cpu",
                    data_signature=postprocessing_fs[i].output_signature,
                    add_batch_dim=False,
                )
            else:
                queue = DynamicShapeQueue(max_size=self.config.queue_sizes["output"])

            self.output_queues.append(queue)

        ### Define the foundation model and the model head nodes
        self.foundation_model = FoundationModel(
            model_card=fm_model_card,
            input_queue=self.input_queue,
            output_queue=self.middle_queue,
            registry=self.registry,
            preprocessing=preprocessing_f,
            config=self.config,
            alias=self.config.foundation_model.get("alias", ""),
            frequency=self.config.foundation_model.get("rate", 1),
        )

        for i, (head, output_queue) in enumerate(zip(head_model_cards, self.output_queues)):
            head_process = ModelHead(
                model_card=head,
                input_queue=self.middle_queue,
                output_queue=output_queue,
                registry=self.registry,
                postprocessing=postprocessing_fs[i],
                config=self.config,
                alias=self.config.model_heads[i].get("alias", ""),
                frequency=self.config.model_heads[i].get("rate", 1),
            )
            self.model_heads.append(head_process)

        # Make sure that output types are correctly specified and can be detected
        for head_id in range(len(self.model_heads)):
            self._detect_output_type(head_id)  # raises NotImplementedError if not implemented

        # Start the nodes
        self.foundation_model.start()
        for head in self.model_heads:
            head.start()

        ### Necessary procedure for CUDAQueue and CUDATimeBuffer to work
        # Share filedescriptors between processes
        self.input_queue.send_shareable_handles([self.foundation_model.pid], False if self.reuse_input_queue_sockets else True)
        self.middle_queue.send_shareable_handles([self.foundation_model.pid] + [head.pid for head in self.model_heads])
        for head, output_queue in zip(self.model_heads, self.output_queues):
            output_queue.send_shareable_handles(head.pid)

        is_loaded = False
        while not is_loaded:
            is_loaded = self.foundation_model.is_loaded() and all([head.is_loaded() for head in self.model_heads])

        # get process names for log analysis
        self.process_names["foundation_model"] = self.foundation_model.name
        self.process_names["model_heads"] = [head.name for head in self.model_heads]

        self.logger.info(MESSAGE.ENGINE_BUILD_SUCCESS)

    def start_inference(self):
        self.foundation_model.start_inference()
        for head in self.model_heads:
            head.start_inference()

    def stop(self):
        if self.foundation_model is not None:
            self.foundation_model.stop()

        for head in self.model_heads:
            if head is not None:
                head.stop()

        if self.input_queue is not None:
            self.input_queue.close()

        if self.middle_queue is not None:
            self.middle_queue.close()

        for output_queue in self.output_queues:
            if output_queue is not None:
                output_queue.close()

        assert mp.active_children() == [], "There are still active child processes after stopping the engine."
        import threading
        assert threading.active_count() == 1, f"There are still active threads after stopping the engine: {list([t.name for t in threading.enumerate()])}"

    def test(self, max_test_time: float = 10.0) -> bool:
        self.logger.info(MESSAGE.ENGINE_TEST_START)
        test_shape = self.config.canonical_image_shape_hwc
        test_inputs = [
            ( np.zeros(test_shape, dtype=np.uint8), -3.01),
            ( 0.5 * np.ones(test_shape, dtype=np.uint8), -2.01),
            ( np.ones(test_shape, dtype=np.uint8), -1.01),
        ]

        start_time = perf_counter()

        for test_input in test_inputs:
            sleep(1)  # first few inferences take longer hence we need to give the system some time
            self.input_image(*test_input)

        for head_idx in range(len(self.output_queues)):
            outputs = []
            while len(outputs) < len(test_inputs) and perf_counter() - start_time < max_test_time:
                try:
                    output_dict = self.get_head_output(head_idx)
                    if output_dict is not None:
                        outputs.append(output_dict)
                except Empty:
                    pass

            if len(outputs) < len(test_inputs):
                self.logger.error(MESSAGE.ENGINE_TEST_FAIL.format(error="Not all outputs were received in time."))
                return False

        ### test commands
        if not self.change_model_rate(self.foundation_model.name, self.config.foundation_model.get("rate", 1)):
            self.logger.error(
                MESSAGE.ENGINE_TEST_FAIL.format(error="Failed to change the frequency of the foundation model.")
            )
            return False

        self.logger.info(MESSAGE.ENGINE_TEST_SUCCESS)
        return True

    def change_model_rate(self, model_name: str, new_rate: float) -> bool:
        """Change the inference rate of the given model. Returns True if successful, False otherwise."""
        target = None
        candidates = [self.foundation_model] + self.model_heads
        for candidate in candidates:
            # check if alias (name of the process) or canonical name (name in model card) matches
            if candidate.name == model_name or candidate.model_card.name == model_name:
                target = candidate
                break

        if not target:
            self.logger.error(MESSAGE.MODEL_NOT_FOUND.format(model_name=model_name))
            return False

        success, error_msg = target.change_frequency(new_rate)
        if not success:
            self.logger.error(error_msg)
            return False
        else:
            self.logger.info(MESSAGE.FREQUENCY_CHANGED.format(model_name=candidate.name, frequency=new_rate))
            return True

    def input_image(self, image: torch.Tensor | np.ndarray, image_id: float = -1) -> bool:
        try:
            self.input_queue.put({PREPROCESSING_INPUT: torch.tensor(image)}, image_id)
            self.logger.info(MESSAGE.ADDED_TO_QUEUE.format(n=image_id, queue_size="N/A"))

        except Full:
            self.logger.warning(MESSAGE.QUEUE_FULL_IMAGE_LOST.format(n=image_id))
            return False

        return True

    def get_raw_head_output(self, head_id: int) -> dict[str, np.ndarray]:
        if head_id >= len(self.output_queues) or head_id < 0:
            self.logger.error(f"Head with id {head_id} does not exist.")
            return {}

        try:
            n, output = self.output_queues[head_id].get_nowait()
            self.logger.info(MESSAGE.OUTPUT_RECEIVED.format(n=n, head_name=self.model_heads[head_id].name))

        except Empty:
            return {}

        return output
    
    def visualize_raw_output(self, head_id: int, raw_output: dict[str, np.ndarray], original_image: None | np.ndarray = None) -> np.ndarray:
        head_cls = getattr(model_architectures, self.model_heads[head_id].model_card.model_class_name)
        return head_cls.visualize_output(raw_output, original_image) 

    def _detect_output_type(self, head_id: int) -> Literal["image", "object_detection"]:
        out_signature = self.model_heads[head_id].postprocessing.output_signature
        signature_shapes = list(out_signature.values())
        if len(out_signature) == 1 and (
            len(signature_shapes[0]) == 2
            or (len(signature_shapes[0]) == 3 and (1 in signature_shapes[0] or 3 in signature_shapes[0]))
        ):
            return "image"
        elif (
            len(out_signature) == 3
            and MH_OBJECT_DETECTION_LABELS in out_signature
            and MH_OBJECT_DETECTION_BOXES_NORMALIZED in out_signature
            and MH_OBJECT_DETECTION_SCORES in out_signature
        ):
            return "object_detection"
        else:
            raise NotImplementedError(
                f"Not able to detect ouput type based on the output signature: \
                                      {out_signature} or output type not implemented."
            )

    def get_head_output(self, head_id: int) -> None | np.ndarray | list[np.ndarray]:
        """Return image or list of arrays based on the output shape of the model head."""
        if head_id >= len(self.output_queues) or head_id < 0:
            self.logger.error(f"Head with id {head_id} does not exist.")
            return None

        raw_output = self.get_raw_head_output(head_id)
        if raw_output == {}:
            return None

        # detect output type
        output_type = self._detect_output_type(head_id)

        if output_type == "image":
            return raw_output[POSTPROCESSING_OUTPUT].numpy()
        elif output_type == "object_detection":
            return [
                raw_output[MH_OBJECT_DETECTION_LABELS].numpy(),
                raw_output[MH_OBJECT_DETECTION_SCORES].numpy(),
                raw_output[MH_OBJECT_DETECTION_BOXES_NORMALIZED].numpy(),
            ]

    def get_foundation_model_params(self) -> dict:
        if self.foundation_model is None or not self.foundation_model.is_alive():
            return {
                "name": self.config.foundation_model.get("alias", self.config.foundation_model.get("canonical_name")),
                "rate": self.config.foundation_model.get("rate"),
            }

        return {"name": self.foundation_model.name, "rate": self.foundation_model.frequency.value}

    def get_model_heads_params(self) -> list[dict]:
        heads = []
        for head_id, head in enumerate(self.model_heads):
            if not head.is_alive():
                heads.append(
                    {
                        "name": self.config.model_heads[head_id].get(
                            "alias", self.config.model_heads[head_id].get("canonical_name")
                        ),
                        "rate": self.config.model_heads[head_id].get("rate"),
                        "output_type": self._detect_output_type(head_id),
                    }
                )
            else:
                heads.append(
                    {"name": head.name, "rate": head.frequency.value, "output_type": self._detect_output_type(head_id)}
                )

        return heads