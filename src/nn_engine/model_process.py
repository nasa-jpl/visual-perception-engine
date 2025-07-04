from typing import Optional, Union
from abc import ABC, abstractmethod
from time import perf_counter, sleep
import signal

import logging
from cuda import cuda
from torch.multiprocessing import Process, Event, Value

from model_architectures.interfaces import ModelInterfaceBase
from model_management.model_cards import ModelCard
from model_management.registry import ModelRegistry
from nn_engine.cuda_buffer import CUDATimeBuffer
from nn_engine.cuda_queue import CUDAQueue
from utils.logging_utils import create_logger, MESSAGE
from nn_engine.config import Config
from nn_engine.cuda_utils import checkCudaErrors


class ModelProcess(Process, ABC):
    def __init__(
        self,
        model_card: ModelCard,
        input_queue: Union[CUDAQueue, CUDATimeBuffer],
        output_queue: Union[CUDAQueue, CUDATimeBuffer],
        registry: ModelRegistry,
        config: Config,
        frequency: float,
        alias: str = "",
        model: Optional[ModelInterfaceBase] = None,
        maximum_sleep_duration: float = 1.0,
        **kwargs,
    ) -> None:
        # Set the name of the process if provided
        if alias:
            kwargs["name"] = alias

        super().__init__(**kwargs)

        self.config = config
        self.registry = registry
        self.frequency = Value("f", frequency)  # In Hz, Shared value for frequency such that it can be easily changed
        self.maximum_sleep_duration = maximum_sleep_duration  # In s, the process will wake up at least every x seconds to reflect changes in its parameters

        self.model = model
        self.model_card = model_card

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.logger: logging.Logger = None  # Will be created in run()
        self.kill_switch = Event()
        self.loaded_flag = Event()
        self.proceed_flag = Event()

    ### BACKEND ###

    def run(self):
        """This method is responsible for communication with outer world, administration and launches the inference procedure whenever needed."""
        # Ignore keyboard interrupts in the model process (which is a child process)
        # Instead they should be stopped by the parent process
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        checkCudaErrors(cuda.cuInit(0))
        self.logger = self.get_logger__()
        self.input_queue.recv_shareable_handles()
        self.output_queue.recv_shareable_handles()
        self.load_model__()
        self.loaded_flag.set()  # Signal that the model has been loaded

        self.proceed_flag.wait()  # Wait for the signal to start inference

        leftover_sleep_time = 0 if self.frequency.value > 0 else self.maximum_sleep_duration
        while not self.kill_switch.is_set():
            if leftover_sleep_time <= 0:
                inference_start_t = perf_counter()
                inference_success = self.inference_procedure__()

                if inference_success:
                    # if the inference was executed successfuly, calculate sleeping duration to match required rate
                    with self.frequency.get_lock():
                        # it is possible that the frequency has been changed during the inference, hence we need to account for that
                        time_interval = (
                            1 / self.frequency.value if self.frequency.value > 0 else self.maximum_sleep_duration
                        )

                    leftover_sleep_time += max(0, time_interval - (perf_counter() - inference_start_t))

                else:
                    # if the inference failed (due to lack of input) we will attempt to perform inference again immediately
                    leftover_sleep_time = 0

            # sleep for the remaining time
            sleep_duration = min(leftover_sleep_time, self.maximum_sleep_duration)
            sleep(sleep_duration)
            leftover_sleep_time -= sleep_duration

            # if the frequency is set to 0, we should sleep for the maximum_sleep_duration
            if self.frequency.value == 0:
                leftover_sleep_time = self.maximum_sleep_duration

    @abstractmethod
    def inference_procedure__(self) -> bool:
        """This method should contain the inference procedure for the model. Should output True if successful, False otherwise."""
        pass

    def load_model__(self):
        if self.model is None:
            self.model = self.registry.load_model_from_card(self.model_card)
            self.logger.info(MESSAGE.MODEL_LOADED.format(model_name=self.model_card.name))

    def get_logger__(self):
        return create_logger(**self.config.logging)

    ### FRONTEND ###

    def is_loaded(self):
        return self.loaded_flag.is_set()

    def start_inference(self):
        self.proceed_flag.set()

    def stop(self):
        """Stop the model process. Should only be called from the parent process. Made to be idempotent."""
        if not self.kill_switch.is_set() and self.is_alive():
            self.kill_switch.set()

            if not self.proceed_flag.is_set():
                self.proceed_flag.set()

            self.join()
            self.close()

    def change_frequency(self, new_frequency: float) -> tuple[bool, str]:
        """Change the frequency of the model process. Returns (True, None) if successful, (False, error_message) otherwise"""

        # check that new frequency is valid
        if new_frequency < 0:
            return False, MESSAGE.INVALID_PARAMETER.format(name="frequency", value=new_frequency, expected=">= 0")

        with self.frequency.get_lock():
            self.frequency.value = new_frequency

        return True, None
