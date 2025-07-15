from queue import Full

import torch

from utils.logging_utils import MESSAGE
from vp_engine.model_process import ModelProcess
from transforms.abstract_transform import AbstractPostprocessing


class ModelHead(ModelProcess):
    def __init__(self, postprocessing: AbstractPostprocessing, **kwargs) -> None:
        super().__init__(**kwargs)
        self.postprocessing = postprocessing
        self.most_recent_timestamp = float("-inf")

    ### BACKEND ###

    def inference_procedure__(self) -> bool:
        with torch.cuda.stream(self.compute_stream):
            out = self.input_queue.get(self.most_recent_timestamp)
            if out is None:
                return False
            n, time_stamp, input_dict = out
            self.logger.info(MESSAGE.IMAGE_RECEIVED.format(n=n))
            self.most_recent_timestamp = time_stamp
            output = self.model.forward_annotated(input_dict)
            self.logger.info(MESSAGE.INFERENCE_COMPLETED.format(n=n))
            output_annotated = self.postprocessing(output)
            try:
                self.output_queue.put_nowait(output_annotated, n)
            except Full:
                self.logger.warning(f"Output queue is full. Dropping output {n}.")
            self.logger.info(
                MESSAGE.OUTPUTS_POSTPROCESSED.format(n=n)
            )  # we call synchronization in queue put_nowait hence only afterwards we can be sure all kernels in this stream have finished

        return True
