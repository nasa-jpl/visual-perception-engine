import os
from typing import Literal, Optional
from dataclasses import dataclass, asdict

from src.nn_engine.naming_convention import *


Precision = Literal["fp32", "fp16", "bfp32", "int8"]
Framework = Literal["tensorrt", "xformers", "torch"]
EncoderSize = Literal["vits", "vitb", "vitl"]


@dataclass(
    kw_only=True
)  # had to set kw_only to True to avoid the issue: https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class ModelCard:
    """Class containing all relevant highlevel details about a particular model"""

    name: str
    precision: Precision
    framework: Framework
    path2weights: str
    n_parameters: int
    model_class_name: str  # the name of the base model class i.e. not TRTModule
    init_arguments: dict
    input_signature: dict[str, tuple]
    output_signature: dict[str, tuple]

    def __post_init__(self):
        self.path2weights = os.path.abspath(self.path2weights)
        assert os.path.isfile(self.path2weights), "The path to the weights is invalid"

        # make sure that the model class name is provided
        assert self.model_class_name, "The model class name must be provided"

        # cast the input and output signatures to a dictionary containing tuples
        self.input_signature = {k: tuple(v) for k, v in self.input_signature.items()}
        self.output_signature = {k: tuple(v) for k, v in self.output_signature.items()}

    def jsonify(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class ModelHeadCard(ModelCard):
    """Class containing all the details about a model head"""

    trained_on: Optional[
        str
    ]  # the name of the model card that this head was trained on, if None, head is a standalone model e.g. doesn't have any parameters


@dataclass(kw_only=True)
class DinoBackend:
    """Class specifying Dino's parameters"""

    encoder: EncoderSize
    n_parameters: int
    features: int
    out_channels: list[int, int, int, int]


@dataclass(kw_only=True)
class DAV2Card(ModelCard):
    """Class containing all the details about a DepthAnythingV2 model"""

    backend: DinoBackend
    model_class_name: str = "DepthAnythingV2"

    def __post_init__(self):
        # allow initializing this class with a dictionary for a backend
        if isinstance(self.backend, dict):
            self.backend = DinoBackend(**self.backend)

        assert isinstance(self.backend, DinoBackend), "The backend must be a DinoBackend object or a dictionary"
