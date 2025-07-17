from queue import Empty

import torch

from vp_engine.model_process import ModelProcess
from utils.logging_utils import MESSAGE
from transforms.abstract_transform import AbstractPreprocessing


class FoundationModel(ModelProcess):
    def __init__(self, preprocessing: AbstractPreprocessing, **kwargs) -> None:
        super().__init__(**kwargs)
        self.preprocessing = preprocessing
        self.most_recent_timestamp = float("-inf")

    ### BACKEND ###

    def inference_procedure__(self) -> bool:
        out = self.input_queue.get(self.most_recent_timestamp)
        if out is None:
            return False
        n, time_stamp, input_dict = out
        self.logger.info(MESSAGE.IMAGE_RECEIVED.format(n=n))
        self.most_recent_timestamp = time_stamp
        preprocessed = self.preprocessing(input_dict)
        self.logger.info(MESSAGE.IMAGE_PREPROCESSED.format(n=n))
        
        if hasattr(self.model, "output_memory_slot"):
            with self.output_queue.writeLock(identifier = n) as output_memory_slot:
                self.model.output_memory_slot = output_memory_slot
                _ = self.model.forward_annotated(preprocessed)
                self.stream.synchronize()
        else:
            output_annotated = self.model.forward_annotated(preprocessed)
            self.output_queue.put(output_annotated, n)
            
        self.logger.info(
            MESSAGE.INFERENCE_COMPLETED.format(n=n)
        )  # we call synchronization in buffer put hence only afterwards we can be sure all kernels in this stream have finished
        return True
