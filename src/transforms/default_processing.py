
import torch

from src.transforms.abstract_transform import AbstractPostprocessing, AbstractPreprocessing
from src.nn_engine.naming_convention import *
from src.nn_engine.shape_utils import assert_correct_io_shapes, assert_correct_types


class DefaultPostprocessing(AbstractPostprocessing):
    """Default postprocessing that adds batch dimension to the output."""

    def __init__(
        self, mh_signature: dict[str, tuple], mh_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        self._type = mh_type
        self._output_signature = {k: tuple(v[1:]) for k, v in mh_signature.items()}
        # remove batch dimension from the input signature
        self._input_signature = mh_signature
        if MH_OUTPUT in self._output_signature:
            self._output_signature[POSTPROCESSING_OUTPUT] = self._output_signature.pop(MH_OUTPUT)

    @assert_correct_types
    @assert_correct_io_shapes
    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove batch dimension from the mh ouput."""
        if MH_OUTPUT in data:
            data[POSTPROCESSING_OUTPUT] = data.pop(MH_OUTPUT)
        return {k: v.squeeze(0) for k, v in data.items()}

    @property
    def input_signature(self) -> dict[str, tuple]:
        return self._input_signature

    @property
    def output_signature(self) -> dict[str, tuple]:
        return self._output_signature

    @property
    def input_type(self) -> torch.dtype:
        return self._type

    @property
    def output_type(self) -> torch.dtype:
        return self._type


class DefaultPreprocessing(AbstractPreprocessing):
    """Dummy processing that does nothing. Used as a filler in the pipeline to make sure that signatures are consistent."""

    def __init__(
        self, fm_signature: dict[str, tuple], fm_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        self._input_shape = (1, *fm_signature[FM_INPUT])
        self._output_shape = fm_signature[FM_INPUT]
        self._type = fm_type

    @assert_correct_types
    @assert_correct_io_shapes
    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {FM_INPUT: data[PREPROCESSING_INPUT].unsqueeze(0)}

    @property
    def input_signature(self) -> dict[str, tuple]:
        return {PREPROCESSING_INPUT: self._input_shape}

    @property
    def output_signature(self) -> dict[str, tuple]:
        return {FM_INPUT: self._output_shape}

    @property
    def input_type(self) -> torch.dtype:
        return self._type

    @property
    def output_type(self) -> torch.dtype:
        return self._type
