import os
from typing import Optional

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from model_architectures.interfaces import ModelInterfaceBase
from model_architectures.dino_foundation_model import DinoFoundationModel
from model_architectures.depth_anything_v2.dpt import DPTHead
from utils.naming_convention import *


class DepthAnythingV2Head(DPTHead, ModelInterfaceBase):
    _is_model_head = True

    available_encoder_sizes = ["vits", "vitb", "vitl"]

    size_specific_configs = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    embed_dim_per_size = {k: v["embed_dim"] for k, v in DinoFoundationModel.size_specific_configs.items()}

    def __init__(
        self, encoder_size: str, max_depth: Optional[float] = None, patch_size: int = 14, img_size: int = 518, **kwargs
    ):
        self.encoder_size = encoder_size
        self.img_size = img_size
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        embed_size = self.embed_dim_per_size[encoder_size]
        batch_size = 1
        self._input_signature = {
            FM_INTERMEDIATE_FEATURES_1: (batch_size, n_patches, embed_size),
            FM_INTERMEDIATE_CLS_TOKEN_1: (
                batch_size,
                embed_size,
            ),
            FM_INTERMEDIATE_FEATURES_2: (batch_size, n_patches, embed_size),
            FM_INTERMEDIATE_CLS_TOKEN_2: (
                batch_size,
                embed_size,
            ),
            FM_INTERMEDIATE_FEATURES_3: (batch_size, n_patches, embed_size),
            FM_INTERMEDIATE_CLS_TOKEN_3: (
                batch_size,
                embed_size,
            ),
            FM_OUTPUT_FEATURES: (batch_size, n_patches, embed_size),
            FM_OUTPUT_CLS_TOKEN: (
                batch_size,
                embed_size,
            ),
        }

        self._output_signature = {MH_OUTPUT: (batch_size, 1, img_size, img_size)}

        assert max_depth is None or max_depth > 0, "The maximum depth must be a positive number"
        self.max_depth = max_depth
        self.metric_depth: bool = bool(max_depth)

        params = self.size_specific_configs[encoder_size].copy()
        params.update(kwargs)

        super(DepthAnythingV2Head, self).__init__(embed_size, **params)

        # DINOv2 foundation model's final norm
        self.norm = torch.nn.LayerNorm(embed_size, eps=1e-6)

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor]]:
        # unflatten the inputs
        it = iter(self.input_signature.keys())
        paitwise_keys = zip(it, it)
        inputs = [(x[key1], x[key2]) for key1, key2 in paitwise_keys]
        return inputs

    def annotate_output(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {MH_OUTPUT: x}

    def forward(self, x: tuple[tuple[torch.Tensor]]) -> torch.Tensor:
        assert len(x) == 4, "The input must be a tuple of 4 tuples"
        assert all(len(x[i]) == 2 for i in range(4)), "Each tuple must contain 2 tensors"
        patch_h = self.img_size // self.patch_size
        patch_w = patch_h

        # Dino foundation model by default outputs unnormalized, not reshaped features, hence
        # we need to do it here to make it compatible
        B, n_patches, embedding_dim = x[0][0].shape
        x = [(self.norm(features).contiguous(), cls_token) for features, cls_token in x]

        depth = super(DepthAnythingV2Head, self).forward(x, patch_h, patch_w)

        if self.max_depth is not None:
            depth = self.max_depth * depth
        else:
            depth = F.relu(depth)
        return depth

    @staticmethod
    def visualize_output(output: dict[str, torch.Tensor], original_image: torch.Tensor = None) -> np.ndarray:
        output = output[POSTPROCESSING_OUTPUT].cpu().squeeze().numpy().astype("uint8")
        colored = cv2.applyColorMap(output, cv2.COLORMAP_INFERNO)
        return colored

    @property
    def input_signature(self) -> dict[str, torch.Tensor]:
        return self._input_signature

    @property
    def output_signature(self) -> dict[str, torch.Tensor]:
        return self._output_signature

    @property
    def is_model_head(self) -> bool:
        return self._is_model_head


if __name__ == "__main__":
    from model_architectures import DinoFoundationModel

    model = DinoFoundationModel("vits", ignore_xformers=True)
    model.load_state_dict(torch.load("models/checkpoints/dinov2_vits14_from_dav2.pth".format(**os.environ)))

    torch.save(
        model.norm.state_dict(),
        "models/checkpoints/final_norm_of_dinov2_vits14_from_dav2.pth".format(**os.environ),
    )
    print("saved norm weights")

    head = DepthAnythingV2Head("vits")
    head.norm = None  # this is not in the original weights so we need to remove it before loading the weights
    head.load_state_dict(
        torch.load("models/checkpoints/depth_anything_v2_head_vits_from_dav2.pth".format(**os.environ))
    )
    head.norm = model.norm

    torch.save(
        head.state_dict(),
        "models/checkpoints/depth_anything_v2_head_vits_from_dav2_with_dino_norm.pth".format(**os.environ),
    )
