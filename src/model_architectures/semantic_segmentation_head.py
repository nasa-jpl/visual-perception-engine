# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
from abc import ABCMeta
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_architectures.interfaces import ModelInterfaceBase
from utils.naming_convention import *
from utils.colormaps import DATASET_COLORMAPS


class BNHead(torch.nn.Module, metaclass=ABCMeta):
    """Just a batchnorm."""

    def __init__(
        self, patch_size: int, img_height_dino: int, img_width_dino: int, embedding_dim: int, num_classes: int
    ):
        super(BNHead, self).__init__()
        self.patch_size = patch_size
        self.img_height_dino = img_height_dino
        self.img_width_dino = img_width_dino
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.bn = nn.SyncBatchNorm(embedding_dim)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.norm = torch.nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, inputs):
        """Forward function."""
        # normalize inputs
        inputs_normalized = self.norm(inputs)
        # reshape inputs
        B = inputs_normalized.shape[0]
        inputs_reshaped = inputs_normalized.reshape(
            B, self.img_height_dino // self.patch_size, self.img_width_dino // self.patch_size, -1
        )
        inputs_permuted = inputs_reshaped.permute(0, 3, 1, 2).contiguous()  # B, H, W, C -> B, C, H, W
        # batchnorm and convolution
        x = self.bn(inputs_permuted)
        x = self.conv_seg(x)
        x = F.interpolate(x, size=(self.img_height_dino, self.img_width_dino), mode="bilinear", align_corners=False)
        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """State dict from the original implementation has different naming convention, hence we need to modify it."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("decode_head."):
                state_dict[key[len("decode_head.") :]] = state_dict[key]
                del state_dict[key]

        super().load_state_dict(state_dict, strict, assign)


class SemanticSegmentationHead(BNHead, ModelInterfaceBase):
    _is_model_head = True

    default_config = {
        "patch_size": 14,  # default patch size for dinov2
        "img_width_dino": 518,
        "img_height_dino": 518,
    }

    size_specific_configs = {
        "vits": {
            "embedding_dim": 384,
        }
    }

    dataset_specific_configs = {
        "voc2012": {
            "num_classes": 21,
        },
        "ade20k": {
            "num_classes": 150,
        },
    }

    def __init__(self, encoder_size: str, dataset: str, **kwargs):
        # Update the default config with the size specific config and the kwargs (highest priority)
        self.cfg = self.default_config.copy()
        self.cfg.update(self.size_specific_configs[encoder_size])
        self.cfg.update(self.dataset_specific_configs[dataset])
        self.cfg.update(kwargs)
        self.dataset = dataset
        num_classes = self.cfg["num_classes"]

        n_patches = (
            self.cfg["img_height_dino"] // self.cfg["patch_size"] * self.cfg["img_width_dino"] // self.cfg["patch_size"]
        )
        batch_size = 1

        self._input_signature = {
            FM_OUTPUT_FEATURES: (batch_size, n_patches, self.cfg["embedding_dim"]),
        }
        self._output_signature = {
            MH_OUTPUT: (batch_size, num_classes, self.cfg["img_height_dino"], self.cfg["img_width_dino"]),
        }
        super().__init__(**self.cfg)

    def annotate_output(self, x: Any) -> dict[str, torch.Tensor]:
        return {MH_OUTPUT: x}

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> Any:
        return x[FM_OUTPUT_FEATURES]

    def forward(self, x: Any) -> Any:
        return BNHead.forward(self, x)

    @staticmethod
    def visualize_output(
        output: dict[str, torch.Tensor], original_image: torch.Tensor = None, dataset: str = None
    ) -> np.ndarray:
        output_logits = output[POSTPROCESSING_OUTPUT].squeeze()

        if dataset is None:
            # we pick voc2012 as default
            dataset = "voc2012"
            assert (output_logits.unique() + 1).max() < SemanticSegmentationHead.dataset_specific_configs["voc2012"][
                "num_classes"
            ], " Number of classes do not match that of in voc2012 dataset, which was chosen by default"

        colormap = DATASET_COLORMAPS[dataset]
        colormap_array = np.array(colormap, dtype=np.uint8)
        segmentation_values = colormap_array[output_logits + 1]
        return segmentation_values

    @property
    def input_signature(self):
        return self._input_signature

    @property
    def output_signature(self):
        return self._output_signature

    @property
    def is_model_head(self):
        return self._is_model_head
