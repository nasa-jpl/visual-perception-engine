import os
import json
import logging
import datetime

import torch
torch._dynamo.config.capture_scalar_outputs = True
from torch2trt import TRTModule

import model_architectures
from model_architectures.custom_trt_module import CustomTRTModule
from utils.naming_convention import *
from model_management import model_cards
from model_management.model_cards import ModelCard
from model_management.util import PRECISION_MAP_TORCH

### setup logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)


class ModelRegistry:
    """Class to track models used in this project along with their properties through a single registry file"""

    def __init__(self, path: str):
        self.path = path

        if not os.path.exists(self.path):
            self.create_empty_registry(self.path)

        assert os.path.isfile(self.path), "The registry file does not exist"

    @staticmethod
    def create_empty_registry(path: str) -> None:
        """Create an empty registry file"""

        assert path.endswith(".jsonl"), "The registry file must be a jsonl file"
        assert not os.path.isfile(path), "The registry file already exists"
        
        # create the directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # create an empty file
        with open(path, "w") as f:
            f.write("")

    def _load_model_card(self, model_dict: dict) -> ModelCard:
        """Load the model dict into a model card
        Args:
            model_dict (dict): dictionary with the model card properties
        Returns:
            ModelCard: model card instance
        Throws:
            AttributeError: if the model type is not recognized
        """
        loaded_model_card = None

        # get the model card class, if missing this will throw attribute error
        card_cls = getattr(model_cards, model_dict["type"])

        loaded_model_card = card_cls(**model_dict["model_card"])

        return loaded_model_card

    def load_registry(self) -> list:
        """Load the registry file"""
        data = []
        with open(self.path) as f:
            for line in f:
                model_dict = json.loads(line)
                data.append(model_dict)
        return data

    def register_model(self, name: str, model: ModelCard) -> None:
        """Register a model in specified jsonl file"""

        new_entry = dict(
            name=name,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
            type=model.__class__.__name__,
            model_card=model.jsonify(),
        )

        # check if the model is already registered
        current_registry_state = self.load_registry()
        for entry in current_registry_state:
            if entry["name"] == new_entry["name"]:
                raise ValueError(f"Model with name '{name}' is already registered")
            if entry["model_card"] == new_entry["model_card"]:
                raise ValueError(f"Model card of {name} is matching with the model card of {entry['name']}")

        if isinstance(model, model_cards.ModelHeadCard):
            # check if the model it depends on is registered
            if model.trained_on not in [entry["name"] for entry in current_registry_state] or model.trained_on is None:
                raise ValueError(f"Model {name} was trained on model {model.trained_on} which is not registered")

        # append the new entry
        with open(self.path, "a") as f:
            f.write(json.dumps(new_entry) + "\n")

        logging.info(f"Model {name} has been successfully registered")

    def get_registered_models(self) -> dict[str, ModelCard]:
        """Get all registered models"""
        models = {model_dict["name"]: self._load_model_card(model_dict) for model_dict in self.load_registry()}
        return models

    @staticmethod
    def load_model_from_card(model_card: ModelCard) -> torch.nn.Module:
        model = None

        for cls in map(model_architectures.__dict__.get, model_architectures.__all__):
            if cls.__name__ == model_card.model_class_name:
                model = cls(**model_card.init_arguments)

        if model is None:
            raise ValueError(
                f"Model {model_card.model_class_name} was not found among the available architectures: {model_architectures.__all__}"
            )

        if model_card.framework == "tensorrt":
            trt_model = TRTModule()
            trt_model.load_state_dict(torch.load(model_card.path2weights))
            model = CustomTRTModule(trt_model, model)
            model.eval()
        else:
            model.load_state_dict(torch.load(model_card.path2weights, map_location="cuda"))
            model = model.to(dtype=PRECISION_MAP_TORCH[model_card.precision], device=torch.device("cuda"))

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if model_card.framework == "torch_compile":
                compiled = torch.compile(model)
                # optimize the model, saving the optimized model is not supported yet, so we need to do it at runtime
                dummy_data_dict = {
                    k: torch.rand((1, *v), dtype=PRECISION_MAP_TORCH[model_card.precision], device=torch.device("cuda"))
                    for k, v in model.input_signature.items()
                }
                _ = compiled(compiled.deannotate_input(dummy_data_dict))
                model = compiled

        return model

    def load_model(self, name: str) -> torch.nn.Module:
        """Load a model available in the registry"""
        try:
            model_card = self.get_registered_models()[name]
        except KeyError:
            raise ValueError(f"Model {name} is not registered")

        model = self.load_model_from_card(model_card)

        return model
