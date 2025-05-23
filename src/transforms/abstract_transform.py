from abc import ABC, abstractmethod
from typing import Optional

import torch


class AbstractTransform(ABC):
    @staticmethod
    @abstractmethod
    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass

    @property
    @abstractmethod
    def input_signature(self) -> dict[str, tuple[Optional[int], Optional[int], Optional[int]]]:
        pass

    @property
    @abstractmethod
    def output_signature(self) -> dict[str, tuple[Optional[int], Optional[int], Optional[int]]]:
        pass

    @property
    @abstractmethod
    def input_type(self) -> torch.dtype:
        pass

    @property
    @abstractmethod
    def output_type(self) -> torch.dtype:
        pass


class AbstractPreprocessing(AbstractTransform):
    @abstractmethod
    def __init__(self, fm_signature: dict[str, tuple], fm_type: torch.dtype, target_height: int, target_width: int):
        """Every preprocessing should receive information about the output type coming from the following foundation model,
        as well as the canonical shape of images used in the engine."""
        pass


class AbstractPostprocessing(AbstractTransform):
    @abstractmethod
    def __init__(
        self, mh_signature: dict[str, tuple], mh_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        """Every postprocessing should receive information about the input type coming from the preceding model head,
        as well as the canonical shape of images used in the engine."""
        pass
