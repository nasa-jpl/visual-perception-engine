import os
import itertools
from typing import Optional

import tensorrt as trt
import torch
from torch2trt import torch2trt

from model_architectures.interfaces import ModelInterfaceBase
import model_architectures
from model_architectures import *
from model_management.model_cards import Precision, Framework, EncoderSize, ModelCard, ModelHeadCard, DAV2Card
from model_management.registry import ModelRegistry
from model_management.util import PRECISION_MAP_TORCH

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(PACKAGE_DIR, "models", "checkpoints")
ENGINE_DIR = os.path.join(PACKAGE_DIR, "models", "engines")
MODEL_REGISTRY_RELATIVE = os.path.join("model_registry", "registry.jsonl")


class ModelExporter:
    """Class used to export a model to TensorRT/ModelCard"""

    # get list of model classes from __all__ in model_architectures
    available_model_types = [getattr(model_architectures, model_class) for model_class in model_architectures.__all__]

    def __init__(self, checkpoint_dir: str, engine_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.engine_dir = engine_dir
        
        assert os.path.exists(self.checkpoint_dir), f"Checkpoint directory {self.checkpoint_dir} does not exist"
        
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir, exist_ok=True)
            print(f"Engine directory {self.engine_dir} created")

    def _compute_n_parameters(self, model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def _model_class_name(model_class: type | str) -> str:
        return model_class if isinstance(model_class, str) else model_class.__name__

    @classmethod
    def get_export_name(
        cls,
        model_class: type | str,
        precision: Precision,
        framework: Framework,
        init_arguments: Optional[dict] = None,
        name_suffix: str = "",
    ) -> str:
        init_arguments = init_arguments or {}
        param_string = "__".join([f"{k}_{v}" for k, v in init_arguments.items()])
        model_class_name = cls._model_class_name(model_class)

        return (
            f"{model_class_name}{name_suffix}_{precision}_{framework}"
            f"{'__' if param_string else ''}{param_string}"
        )

    @staticmethod
    def get_dav2_export_name(
        encoder_size: EncoderSize,
        precision: Precision,
        framework: Framework,
        init_arguments: Optional[dict] = None,
    ) -> str:
        init_arguments = init_arguments or {}
        is_metric = bool(init_arguments.get("max_depth", None))

        return (
            f"DepthAnythingV2_{encoder_size}_{framework}_{precision}"
            f"{'_metric' if is_metric else ''}"
        )

    def _initialize_model_class(self, model_class: type | str, init_arguments: dict) -> ModelInterfaceBase:
        """Initialize the model class and return it"""
        if isinstance(model_class, str):
            assert model_class in model_architectures.__all__, (
                f"Model class {model_class} not found in available models"
            )
            model_class = getattr(model_architectures, model_class)

        assert model_class in self.available_model_types, f"Model class {model_class} not found in available models"
        model = model_class(**init_arguments)

        return model

    def _get_model_from_class(
        self, model_class: type | str, precision: Precision, init_arguments: dict, weights_path: str
    ) -> ModelInterfaceBase:
        model = self._initialize_model_class(model_class, init_arguments)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, weights_path), map_location="cuda", weights_only=False))

        for param in model.parameters():
            param.requires_grad = False

        model = model.eval().to(dtype=PRECISION_MAP_TORCH[precision], device=torch.device("cuda"))
        return model

    def export2trt(self, precision: Precision, model: ModelInterfaceBase) -> None:
        dummy_data_dict = {
            k: torch.ones(*v, dtype=PRECISION_MAP_TORCH[precision], device=torch.device("cuda"))
            for k, v in model.input_signature.items()
        }
        dummy_data = model.deannotate_input(dummy_data_dict)

        trt_model = torch2trt(
            model,
            [dummy_data],
            input_names=list(model.input_signature.keys()),
            output_names=list(model.output_signature.keys()),
            fp16_mode=precision == "fp16",
            int8_mode=precision == "int8",
            log_level=trt.Logger.INFO,
            use_onnx=True,
        )

        return trt_model

    def check_if_compileable(self, precision: Precision, model: ModelInterfaceBase, batch_size: int = 1) -> bool:
        """Check if the model can be compiled using torch.compile"""
        try:
            compiled = torch.compile(model)
            dummy_data_dict = {
                k: torch.ones((batch_size, *v), dtype=PRECISION_MAP_TORCH[precision], device=torch.device("cuda"))
                for k, v in model.input_signature.items()
            }
            _ = compiled(compiled.deannotate_input(dummy_data_dict))
            return True
        except Exception as e:
            print(f"Model {model.__class__.__name__} cannot be compiled: {e}")
            return False

    def export_from_trt_engine(
        self,
        precision: Precision,
        model_class: type | str,
        engine_path: str,
        trained_on: Optional[str] = None,
        name_suffix: str = "",  # suffix to the name of the model in case of multiple models with the same name but different weights
        n_parameters: int = -1,  # number of parameters in the model
    ) -> tuple[str, ModelCard]:
        model = self._initialize_model_class(model_class, {})

        name = f"{model_class.__name__}{name_suffix}_{precision}"

        common_model_card_params = dict(
            name=name,
            precision=precision,
            framework="tensorrt",
            path2weights=engine_path,
            n_parameters=n_parameters,
            model_class_name=self._model_class_name(model_class),
            init_arguments={},
            input_signature=model.input_signature,
            output_signature=model.output_signature,
        )

        if model.is_model_head:
            model_card = ModelHeadCard(trained_on=trained_on, **common_model_card_params)
        else:
            model_card = ModelCard(**common_model_card_params)

        return name, model_card

    def export(
        self,
        precision: Precision,
        framework: Framework,
        model_class: type | str,
        weights_path: str,
        init_arguments: dict = {},
        trained_on: Optional[str] = None,
        name_suffix: str = "",  # suffix to the name of the model in case of multiple models with the same name but different weights
    ) -> tuple[str, ModelCard]:
        # Load the model
        model = self._get_model_from_class(model_class, precision, init_arguments, weights_path)
        n_parameters = self._compute_n_parameters(model)

        name = self.get_export_name(
            model_class=model_class,
            precision=precision,
            framework=framework,
            init_arguments=init_arguments,
            name_suffix=name_suffix,
        )
        path = os.path.join(self.checkpoint_dir, weights_path)

        if framework == "tensorrt":
            # Convert the model to TensorRT
            trt_model = self.export2trt(precision, model)
            path = os.path.join(self.engine_dir, f"{weights_path.split('.')[0]}_{precision}.pth")
            torch.save(trt_model.state_dict(), path)
            print(f"Model {name} has been successfully exported to TensorRT")

        common_model_card_params = dict(
            name=name,
            precision=precision,
            framework=framework,
            path2weights=path,
            n_parameters=n_parameters,
            model_class_name=self._model_class_name(model_class),
            init_arguments=init_arguments,
            input_signature=model.input_signature,
            output_signature=model.output_signature,
        )

        if model.is_model_head:
            model_card = ModelHeadCard(trained_on=trained_on, **common_model_card_params)
        else:
            model_card = ModelCard(**common_model_card_params)

        return name, model_card

    def export_dav2(
        self,
        precision: Precision,
        framework: Framework,
        encoder_size: EncoderSize,
        weights_path: str,
        init_arguments: dict = {},
    ) -> tuple[str, DAV2Card]:
        # make sure that init_arguments passed have highest priority (i.e. cannot be overwritten)
        is_metric = bool(init_arguments.get("max_depth", None))
        init_arguments = {
            **DepthAnythingV2Head.size_specific_configs[encoder_size],
            "encoder": encoder_size,
            "ignore_xformers": framework != "xformers",
            **init_arguments,
        }

        # Load the model
        model = self._get_model_from_class(DepthAnythingV2, precision, init_arguments, weights_path)

        n_parameters_dino = self._compute_n_parameters(model.pretrained)
        n_parameters = self._compute_n_parameters(model)

        name = self.get_dav2_export_name(
            encoder_size=encoder_size,
            precision=precision,
            framework=framework,
            init_arguments=init_arguments,
        )
        path = os.path.join(self.checkpoint_dir, weights_path)

        if framework == "tensorrt":
            # Convert the model to TensorRT
            trt_model = self.export2trt(precision, model)
            path = os.path.join(self.engine_dir, weights_path)
            torch.save(trt_model.state_dict(), path)
            print(f"Model {name} has been successfully exported to TensorRT")
        elif framework == "torch_compile":
            if not self.check_if_compileable(model, path):
                raise ValueError(f"Model {name} cannot be compiled to TorchScript")

        model_card = DAV2Card(
            name=name,
            precision=precision,
            framework=framework,
            path2weights=path,
            n_parameters=n_parameters,
            init_arguments=init_arguments,
            input_signature=model.input_signature,
            output_signature=model.output_signature,
            backend={
                **DepthAnythingV2Head.size_specific_configs[encoder_size],
                "encoder": encoder_size,
                "n_parameters": n_parameters_dino,
            },
        )

        return name, model_card


def export_default_models(registry_path=MODEL_REGISTRY_RELATIVE):
    """Export default models such that they can be used in the engine.

    This function is resumable: if a model is already present in the registry,
    it is skipped instead of being re-exported.
    """
    registry = ModelRegistry(os.path.join(PACKAGE_DIR, registry_path))
    exporter = ModelExporter(CHECKPOINT_DIR, ENGINE_DIR)

    registered_models = registry.get_registered_models()

    def is_already_registered(name: str) -> bool:
        if name in registered_models:
            print(f"Skipping {name}: already in registry")
            return True
        return False

    def register_exported_model(expected_name: str, exported_name: str, model_card: ModelCard):
        if exported_name != expected_name:
            raise ValueError(
                f"Exported model name mismatch. Expected '{expected_name}', got '{exported_name}'"
            )

        registry.register_model(exported_name, model_card)
        registered_models[exported_name] = model_card

    ### export DAV2 model
    iterator = itertools.product(["vits"], ["fp16"], ["torch", "tensorrt"], [False])
    for encoder_size, precision, framework, metric in iterator:
        init_arguments = {"max_depth": 80 if metric else None}

        expected_name = exporter.get_dav2_export_name(
            encoder_size=encoder_size,
            precision=precision,
            framework=framework,
            init_arguments=init_arguments,
        )

        if is_already_registered(expected_name):
            continue

        name, model_card = exporter.export_dav2(
            precision,
            framework,
            encoder_size=encoder_size,
            init_arguments=init_arguments,
            weights_path=f"depth_anything_v2_{'metric_vkitti_' if metric else ''}{encoder_size}.pth",
        )

        register_exported_model(expected_name, name, model_card)

    ### export DAV2 version of DINOv2 and its depth head
    iterator = itertools.product(["vits"], ["fp16"], ["torch", "tensorrt"])
    for encoder_size, precision, framework in iterator:
        fn_init_arguments = {
            "encoder_size": encoder_size,
            "ignore_xformers": framework != "xformers",
        }

        fn_name = exporter.get_export_name(
            model_class=DinoFoundationModel,
            precision=precision,
            framework=framework,
            init_arguments=fn_init_arguments,
            name_suffix="_from_dav2",
        )

        if not is_already_registered(fn_name):
            name, model_card = exporter.export(
                precision,
                framework,
                DinoFoundationModel,
                init_arguments=fn_init_arguments,
                name_suffix="_from_dav2",
                weights_path=f"dinov2_{encoder_size}14_from_dav2.pth",
            )

            register_exported_model(fn_name, name, model_card)

        # export depth model heads
        head_init_arguments = {"encoder_size": encoder_size}

        head_name = exporter.get_export_name(
            model_class=DepthAnythingV2Head,
            precision=precision,
            framework=framework,
            init_arguments=head_init_arguments,
        )

        if not is_already_registered(head_name):
            name, model_card = exporter.export(
                precision,
                framework,
                DepthAnythingV2Head,
                init_arguments=head_init_arguments,
                weights_path=f"depth_anything_v2_head_{encoder_size}_from_dav2_with_dino_norm.pth",
                trained_on=fn_name,
            )

            register_exported_model(head_name, name, model_card)

    ### export DINOv2 foundation model and respective heads
    iterator = itertools.product(["vits"], ["fp16"], ["torch", "tensorrt"])
    for encoder_size, precision, framework in iterator:
        fn_init_arguments = {
            "encoder_size": encoder_size,
            "ignore_xformers": framework != "xformers",
        }

        fn_name = exporter.get_export_name(
            model_class=DinoFoundationModel,
            precision=precision,
            framework=framework,
            init_arguments=fn_init_arguments,
        )

        if not is_already_registered(fn_name):
            name, model_card = exporter.export(
                precision,
                framework,
                DinoFoundationModel,
                init_arguments=fn_init_arguments,
                weights_path=f"dinov2_{encoder_size}14_pretrain.pth",
            )

            register_exported_model(fn_name, name, model_card)

        # export depth model heads
        for dataset in ["kitti", "nyu"]:
            dpt_init_arguments = {"encoder_size": encoder_size}

            dpt_name = exporter.get_export_name(
                model_class=DPTHead,
                precision=precision,
                framework=framework,
                init_arguments=dpt_init_arguments,
                name_suffix=f"_{dataset}",
            )

            if is_already_registered(dpt_name):
                continue

            name, model_card = exporter.export(
                precision,
                framework,
                DPTHead,
                init_arguments=dpt_init_arguments,
                weights_path=f"dinov2_{encoder_size}14_{dataset}_dpt_head.pth",
                name_suffix=f"_{dataset}",
                trained_on=fn_name,
            )

            register_exported_model(dpt_name, name, model_card)

        # export object detection model heads
        if framework != "tensorrt":
            obj_init_arguments = {"encoder_size": encoder_size}

            obj_name = exporter.get_export_name(
                model_class=ObjectDetectionHead,
                precision=precision,
                framework=framework,
                init_arguments=obj_init_arguments,
            )

            if not is_already_registered(obj_name):
                name, model_card = exporter.export(
                    precision,
                    framework,
                    ObjectDetectionHead,
                    init_arguments=obj_init_arguments,
                    weights_path=f"object_detection_head_{encoder_size}.pth",
                    trained_on=fn_name,
                )

                register_exported_model(obj_name, name, model_card)

        # export semantic segmentation model heads
        for dataset in ["voc2012", "ade20k"]:
            semseg_init_arguments = {
                "encoder_size": encoder_size,
                "dataset": dataset,
            }

            semseg_name = exporter.get_export_name(
                model_class=SemanticSegmentationHead,
                precision=precision,
                framework=framework,
                init_arguments=semseg_init_arguments,
            )

            if is_already_registered(semseg_name):
                continue

            name, model_card = exporter.export(
                precision,
                framework,
                SemanticSegmentationHead,
                init_arguments=semseg_init_arguments,
                weights_path=f"semantic_segmentation_head_{dataset}_{encoder_size}.pth",
                trained_on=fn_name,
            )

            register_exported_model(semseg_name, name, model_card)

