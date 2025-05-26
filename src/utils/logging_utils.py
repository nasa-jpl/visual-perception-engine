import logging
from enum import Enum
from typing import Optional

import torch.multiprocessing as mp


class MESSAGE(Enum):
    """Enum class containing messages used for logging.
    It is used to standardize the logging messages to make log analysis easier."""

    IMAGE_RECEIVED = "Received input image {n}"
    INPUT_MOVED_TO_GPU = "Moved input for image {n} to GPU"
    OUTPUT_MOVED_TO_CPU = "Moved output for image {n} to CPU"
    IMAGE_PREPROCESSED = "Preprocessed image {n}"
    INFERENCE_COMPLETED = "Inference on image {n} completed"
    MODEL_LOADED = "Model {model_name} loaded successfully"
    MAX_QUEUE_SIZE_EXCEEDED = "Maximum queue size exceeded. Removed item {n} from queue"
    QUEUE_FULL_IMAGE_LOST = "Queue is full. Failed to add item {n}"
    ADDED_TO_QUEUE = "Added item {n} to queue. Queue size: {queue_size}"
    ENGINE_TEST_START = "Engine test started"
    ENGINE_TEST_SUCCESS = "Engine test successful"
    ENGINE_TEST_FAIL = "Engine test failed: {error}"
    ENGINE_BUILD_START = "Building engine with foundation model {fm_model_name} and {num_heads} model heads"
    ENGINE_BUILD_SUCCESS = "Engine was built successfully"
    OUTPUT_RECEIVED = "Received output for image {n} from head {head_name}"
    OUTPUTS_POSTPROCESSED = "Postprocessed outputs for image {n}"
    OUTPUT_CLONED = "Cloned output for image {n} from head {head_name}"
    SAVING_OUTPUTS = "Saving outputs of the engine run"
    OUTPUTS_SAVED = "Outputs saved successfully"
    FREQUENCY_CHANGED = "Frequency of {model_name} changed to {frequency} Hz"
    INVALID_PARAMETER = "Invalid value {value} for parameter {name}. Expected: {expected}"
    MODEL_NOT_FOUND = "Model {model_name} not found"

    def __str__(self) -> str:
        """Adds the message ID to the beginning of the message to make log analysis easier."""
        return f"|{self.get_id()}| {self.value}"

    def get_id(self) -> int:
        return self.names().index(self.name)

    def format(self, **kwargs) -> str:
        """Format the message with the given keyword arguments."""
        return str(self).format(**kwargs)

    def names(self) -> list[str]:
        return list(self.__class__.__members__)


def create_logger(
    log_to_console: bool = True, log_file: Optional[str] = None, log_level=logging.INFO
) -> logging.Logger:
    """Create a logger for the current process.

    :param log_file: Optional, the path to the log file.
    :param log_to_console: Optional, whether to log to stdout.

    :return: The logger object.
    """
    logger = mp.get_logger()
    logger.setLevel(log_level)
    formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(processName)s] %(message)s")

    # Clean up the handlers
    # this bit will make sure you won't have
    # duplicated messages in the output
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Add the handlers
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if log_to_console:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    return logger
