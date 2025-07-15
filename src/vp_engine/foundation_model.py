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
        copy_done_evt = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(self.copy_stream):
            out = self.input_queue.get(self.most_recent_timestamp)
            if out is None:
                return False
            n, time_stamp, input_dict = out
            self.logger.info(MESSAGE.IMAGE_RECEIVED.format(n=n))
            self.most_recent_timestamp = time_stamp
            copy_done_evt.record()
            
        compute_done_evt = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(copy_done_evt)
            preprocessed = self.preprocessing(input_dict)
            self.logger.info(MESSAGE.IMAGE_PREPROCESSED.format(n=n))
            with self.output_queue.writeLock(identifier = n) as output_memory_slot:
                if hasattr(self.model, "output_memory_slot"):
                    self.model.output_memory_slot = output_memory_slot
                output_annotated = self.model.forward_annotated(preprocessed)
                compute_done_evt.record()
                self.copy_stream.wait_event(compute_done_evt)
        self.logger.info(
            MESSAGE.INFERENCE_COMPLETED.format(n=n)
        )  # we call synchronization in buffer put hence only afterwards we can be sure all kernels in this stream have finished
        return True
