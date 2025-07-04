
from transforms.abstract_transform import AbstractPostprocessing
from utils.naming_convention import *
from utils.shape_utils import assert_correct_io_shapes, assert_correct_types
import torch
import torch.nn.functional as F


class SemSegPostprocessing(AbstractPostprocessing):
    """Transform tensor (float) to uint8 tensor, without any scaling."""

    def __init__(
        self, mh_signature: dict[str, tuple], mh_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        self.batch_size = mh_signature[MH_OUTPUT][0]
        self.num_classes = mh_signature[MH_OUTPUT][1]
        self.input_height = mh_signature[MH_OUTPUT][-2]
        self.input_width = mh_signature[MH_OUTPUT][-1]
        self.output_height = canonical_height
        self.output_width = canonical_width
        self._input_type = mh_type
        self._output_type = torch.uint8  # uint8 is the default type for images

    @assert_correct_types
    @assert_correct_io_shapes
    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = data[MH_OUTPUT]
        # interpolate to original size and choose one class per pixel
        output = F.interpolate(x, size=(self.output_height, self.output_width), mode="bilinear", align_corners=False)
        output = F.softmax(output, dim=1).argmax(dim=1, keepdim=True).squeeze(0).to(torch.uint8).permute(1, 2, 0)
        return {POSTPROCESSING_OUTPUT: output}

    @property
    def input_signature(self) -> dict[str, tuple]:
        return {MH_OUTPUT: (self.batch_size, self.num_classes, self.input_height, self.input_width)}

    @property
    def output_signature(self) -> dict[str, tuple]:
        return {POSTPROCESSING_OUTPUT: (self.output_height, self.output_width, 1)}

    @property
    def input_type(self) -> torch.dtype:
        return self._input_type

    @property
    def output_type(self) -> torch.dtype:
        return self._output_type
