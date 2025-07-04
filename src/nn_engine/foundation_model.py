from queue import Empty

from nn_engine.model_process import ModelProcess
from utils.logging_utils import MESSAGE
from transforms.abstract_transform import AbstractPreprocessing


class FoundationModel(ModelProcess):
    def __init__(self, preprocessing: AbstractPreprocessing, **kwargs) -> None:
        super().__init__(**kwargs)
        self.preprocessing = preprocessing

    ### BACKEND ###

    def inference_procedure__(self) -> bool:
        try:
            n, input = self.input_queue.get_nowait()
        except Empty:
            return False
        self.logger.info(MESSAGE.IMAGE_RECEIVED.format(n=n))
        preprocessed = self.preprocessing(input)
        self.logger.info(MESSAGE.IMAGE_PREPROCESSED.format(n=n))
        output_annotated = self.model.forward_annotated(preprocessed)
        self.output_queue.put(output_annotated, n)
        self.logger.info(
            MESSAGE.INFERENCE_COMPLETED.format(n=n)
        )  # we call synchronization in buffer put hence only afterwards we can be sure all kernels in this stream have finished
        return True
