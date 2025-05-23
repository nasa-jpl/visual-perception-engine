from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np

from src.nn_engine.shape_utils import assert_correct_io_shapes


class ModelInterfaceBase(ABC):
    """Minimal interface for all models in the perception engine."""

    @property
    @abstractmethod
    def input_signature(self) -> dict[str, tuple[int]]:
        # The input signature should be a dictionary with the keys being the names of the inputs
        # and the values being tuples of integers representing the shape of the input.
        pass

    @property
    @abstractmethod
    def output_signature(self) -> dict[str, tuple[int]]:
        # The output signature should be a dictionary with the keys being the names of the outputs
        # and the values being tuples of integers representing the shape of the output.
        pass

    @property
    @abstractmethod
    def is_model_head(self) -> bool:
        # The is_model_head property should return True if the model is a head model, False otherwise.
        pass

    # NOTE distinction into annotated and non-annotated forward methods is
    # necessary because torch2trt does not support forward with dicts

    @abstractmethod
    def forward(self, *x: Any) -> Any:
        """Forward pass of the model."""
        pass

    # NOTE deannotate_input and annotate_output methods are necessary for annotation handling
    # when using TRTModule

    @abstractmethod
    def deannotate_input(self, x: dict[str, torch.Tensor]) -> Any:
        """Remove the annotations from the input tensors.
        Converts the annotated dictionary input to the format accepted by forward function."""
        pass

    @abstractmethod
    def annotate_output(self, x: Any) -> dict[str, torch.Tensor]:
        """Annotate the output tensors with the output signature.
        Converts the output of the forward function to a dictionary with annotated keys."""
        pass

    @assert_correct_io_shapes
    def forward_annotated(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the model with annotated inputs and outputs."""
        deannotated_input = self.deannotate_input(x)
        output = self.forward(deannotated_input)
        return self.annotate_output(output)

    @staticmethod
    def visualize_output(output: dict[str, torch.Tensor], original_image: torch.Tensor) -> np.ndarray:
        """Visualize the output of the model as an image.
        NOTE: this method is optional and should be implemented only if it makes sense to represent model output as an image
        """
        raise NotImplementedError()
