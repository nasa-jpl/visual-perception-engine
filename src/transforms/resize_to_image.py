
from src.transforms.abstract_transform import AbstractPostprocessing
from src.nn_engine.naming_convention import *
from src.nn_engine.shape_utils import assert_correct_io_shapes, assert_correct_types
import torch
import torch.nn.functional as F


class ResizeAndToCV2Image(AbstractPostprocessing):
    """Transform tensor (float) to numpy image (uint8) of a desired shape."""

    def __init__(
        self, mh_signature: dict[str, tuple], mh_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        self.input_channels = mh_signature[MH_OUTPUT][-3]
        self.input_height = mh_signature[MH_OUTPUT][-1]
        self.input_width = mh_signature[MH_OUTPUT][-2]
        self.output_height = canonical_height
        self.output_width = canonical_width
        self._input_type = mh_type
        self._output_type = torch.uint8  # uint8 is the default type for images

    @assert_correct_types
    @assert_correct_io_shapes
    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data = F.interpolate(
            data[MH_OUTPUT], (self.output_height, self.output_width), mode="bilinear", align_corners=False
        )
        data = data.squeeze(0)
        img = (data - data.min()) / (data.max() - data.min()) * 255.0
        return {POSTPROCESSING_OUTPUT: img.to(torch.uint8).permute(1, 2, 0)}

    @property
    def input_signature(self) -> dict[str, tuple]:
        batch_size = 1
        return {MH_OUTPUT: (batch_size, self.input_channels, self.input_height, self.input_width)}

    @property
    def output_signature(self) -> dict[str, tuple]:
        return {POSTPROCESSING_OUTPUT: (self.output_height, self.output_width, self.input_channels)}

    @property
    def input_type(self) -> torch.dtype:
        return self._input_type

    @property
    def output_type(self) -> torch.dtype:
        return self._output_type
