from functools import partial
from itertools import chain

import torch

from model_architectures.interfaces import ModelInterfaceBase
from model_architectures.depth_anything_v2.dinov2 import DinoVisionTransformer
from model_architectures.depth_anything_v2.dinov2_layers import MemEffAttention, NestedTensorBlock as Block
from utils.naming_convention import *


class DinoFoundationModel(DinoVisionTransformer, ModelInterfaceBase):
    """Class for the Dino foundation model fullfiling the requirements of the ModelInterfaceBase"""

    _is_model_head = False
    available_sizes = ["vits", "vitb", "vitl"]

    intermediate_layer_idx = {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23],
        "vitg": [9, 19, 29, 39],
    }

    default_config = {  # default based on the DepthAnythingV2 implementation
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "block_chunks": 0,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
    }

    size_specific_configs = {
        "vits": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4,
            "block_fn": partial(Block, attn_class=MemEffAttention),
            "num_register_tokens": 0,
        },
        "vitb": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "block_fn": partial(Block, attn_class=MemEffAttention),
            "num_register_tokens": 0,
        },
        "vitl": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
            "block_fn": partial(Block, attn_class=MemEffAttention),
            "num_register_tokens": 0,
        },
        "vitg": {
            "embed_dim": 1536,
            "depth": 40,
            "num_heads": 24,
            "mlp_ratio": 4,
            "block_fn": partial(Block, attn_class=MemEffAttention),
            "num_register_tokens": 0,
        },
    }

    def __init__(
        self,
        encoder_size: str,
        ignore_xformers: bool = True,
        apply_final_norm: bool = False,
        reshape_to_patches: bool = False,
        **kwargs,
    ):
        self.size = encoder_size

        self.apply_final_norm = (
            apply_final_norm  # Depth anything V2 does apply final norm whereas depth head from DINOv2 does not
        )
        self.reshape_to_patches = (
            reshape_to_patches  # Depth anything V2 does not reshape to patches whereas depth head from DINOv2 does
        )

        self.default_config["ignore_xformers"] = ignore_xformers
        self.default_config["ffn_layer"] = "mlp" if encoder_size != "vitg" else "swiglufused"

        # Update the default config with the size specific config and the kwargs (highest priority)
        params = self.default_config.copy()
        params.update(self.size_specific_configs[encoder_size])
        params.update(kwargs)

        # Set the input and output signatures
        img_size = params["img_size"]
        self.patch_h = img_size // params["patch_size"]
        self.patch_w = img_size // params["patch_size"]
        n_patches = self.patch_h * self.patch_w
        embed_size = params["embed_dim"]
        batch_size = 1
        intermediate_features_shape = (
            (batch_size, embed_size, self.patch_w, self.patch_h)
            if self.reshape_to_patches
            else (batch_size, n_patches, embed_size)
        )

        self._input_signature = {FM_INPUT: (batch_size, 3, img_size, img_size)}
        self._output_signature = {
            FM_INTERMEDIATE_FEATURES_1: intermediate_features_shape,
            FM_INTERMEDIATE_CLS_TOKEN_1: (
                batch_size,
                embed_size,
            ),
            FM_INTERMEDIATE_FEATURES_2: intermediate_features_shape,
            FM_INTERMEDIATE_CLS_TOKEN_2: (
                batch_size,
                embed_size,
            ),
            FM_INTERMEDIATE_FEATURES_3: intermediate_features_shape,
            FM_INTERMEDIATE_CLS_TOKEN_3: (
                batch_size,
                embed_size,
            ),
            FM_OUTPUT_FEATURES: intermediate_features_shape,
            FM_OUTPUT_CLS_TOKEN: (
                batch_size,
                embed_size,
            ),
        }

        super(DinoFoundationModel, self).__init__(**params)

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return x[FM_INPUT]

    def annotate_output(self, x: tuple[tuple[torch.Tensor]]) -> dict[str, torch.Tensor]:
        # flatten the outputs
        outputs = list(chain(*x))
        return {key: outputs[i] for i, key in enumerate(self.output_signature.keys())}

    def forward(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor]]:
        return self.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.size],
            return_class_token=True,
            reshape=self.reshape_to_patches,
            norm=self.apply_final_norm,
        )

    @property
    def input_signature(self):
        return self._input_signature

    @property
    def output_signature(self):
        return self._output_signature

    @property
    def is_model_head(self):
        return self._is_model_head


if __name__ == "__main__":
    model = DinoFoundationModel("vits", ignore_xformers=True)
