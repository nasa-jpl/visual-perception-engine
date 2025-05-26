from typing import Any, OrderedDict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.utils import draw_bounding_boxes

from transforms.dinov2_preprocessing import DINOV2PreprocessingTorch
from model_architectures.dino_foundation_model import DinoFoundationModel
from utils.naming_convention import *

from model_architectures.interfaces import ModelInterfaceBase


def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class CustomDinoObjectDet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        out_channels: int,
        img_height_dino: int,
        img_width_dino: int,
        patch_size: int,
        processing_height: int,
        processing_width: int,
    ):
        super().__init__()
        self.img_height_dino = img_height_dino
        self.img_width_dino = img_width_dino
        self.patch_size = patch_size
        self.processing_height = processing_height
        self.processing_width = processing_width

        self.norm = torch.nn.LayerNorm(embedding_dim, eps=1e-6)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up_8 = convrelu(in_channels=embedding_dim, out_channels=out_channels, kernel=3, stride=1, padding=1)
        self.conv_up_4 = convrelu(in_channels=out_channels, out_channels=out_channels, kernel=3, stride=1, padding=1)
        self.conv_down_16 = convrelu(in_channels=out_channels, out_channels=out_channels, kernel=3, stride=2, padding=1)
        self.conv_down_32 = convrelu(in_channels=out_channels, out_channels=out_channels, kernel=3, stride=2, padding=1)
        self.conv_down_64 = convrelu(in_channels=out_channels, out_channels=out_channels, kernel=3, stride=2, padding=1)

    def forward(self, dino_features):
        feats_normalized = self.norm(dino_features)

        B = feats_normalized.shape[0]
        feats_reshaped = feats_normalized.reshape(
            B, self.img_height_dino // self.patch_size, self.img_width_dino // self.patch_size, -1
        )
        feats_permuted = feats_reshaped.permute(0, 3, 1, 2).contiguous()  # B, H, W, C -> B, C, H, W

        new_feat_height, new_feat_width = int(self.processing_height / 8), int(self.processing_width / 8)
        dino_features = F.interpolate(
            feats_permuted, [new_feat_height, new_feat_width], mode="bilinear", align_corners=True
        )

        feat_8 = self.conv_up_8(dino_features)
        feat_4_up = self.upsample(feat_8)
        feat_4 = self.conv_up_4(feat_4_up)
        feat_16 = self.conv_down_16(feat_8)
        feat_32 = self.conv_down_32(feat_16)
        feat_64 = self.conv_down_64(feat_32)

        out = {"0": feat_4, "1": feat_8, "2": feat_16, "3": feat_32, "pool": feat_64}
        return out


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class ObjectDetector(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        out_channels: int,
        embedding_dim: int,
        conf_thres: float,
        use_upsampler: bool,
        min_size: int,
        max_size: int,
        box_nms_thresh: float,
        processing_height: int,
        processing_width: int,
        img_width_dino: int,
        img_height_dino: int,
    ):
        super().__init__()
        self.num_classes = num_classes  # background, backpack, stairs
        self.conf_thres = conf_thres  # 0.750  # confidence threshold
        self.processing_height = processing_height
        self.processing_width = processing_width
        self.img_width_dino = img_width_dino
        self.img_height_dino = img_height_dino
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.use_upsampler = use_upsampler
        self.min_size = min_size
        self.max_size = max_size
        self.box_nms_thresh = box_nms_thresh

        backbone = CustomDinoObjectDet(
            self.embedding_dim,
            self.out_channels,
            self.img_height_dino,
            self.img_width_dino,
            self.patch_size,
            self.processing_height,
            self.processing_width,
        )
        backbone.out_channels = self.out_channels

        anchor_sizes = (
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"], output_size=7, sampling_ratio=2
        )

        resolution = roi_pooler.output_size[0]
        representation_size = 512
        box_head = TwoMLPHead(self.out_channels * resolution**2, representation_size)

        representation_size = 512
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        self.model = FasterRCNN(
            backbone,
            min_size=self.min_size,
            max_size=self.max_size,
            box_head=box_head,
            box_predictor=box_predictor,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_nms_thresh=self.box_nms_thresh,
        )

        def new_forward(images, targets=None):
            features = self.model.backbone(images)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            processing_shape = (self.processing_height, self.processing_width)

            images = ImageList(
                torch.zeros(1, 3, *processing_shape), [processing_shape]
            )  # we need to use this because rpn internally uses image sizes to calculate anchors

            proposals, _ = self.model.rpn(images, features, targets)
            detections, _ = self.model.roi_heads(features, proposals, [processing_shape], targets)
            detections = self.model.transform.postprocess(detections, [processing_shape], [processing_shape])

            return self.model.eager_outputs({}, detections)

        self.model.forward = new_forward

    def forward(self, x):
        predictions = self.model(x)
        assert len(predictions) == 1, "Expected only one prediction, got more"
        pred = predictions[0]
        det_idx = torch.where(pred["scores"] > self.conf_thres)[0]
        # add batch dimension via [None]
        pred["labels"] = pred["labels"][det_idx][None].to(torch.float16)
        pred["scores"] = pred["scores"][det_idx][None]
        pred["boxes"] = pred["boxes"][det_idx][None]
        # scale boxes to [0, 1]
        pred["boxes"] /= torch.tensor(
            [self.processing_width, self.processing_height, self.processing_width, self.processing_height],
            device=pred["boxes"].device,
        )
        return pred

    @staticmethod
    def label_to_cat(label):
        label_to_cat = {1: "backpack", 2: "stairs"}  # we are shifting the classes by 1 later on
        return label_to_cat[label]


class ObjectDetectionHead(ModelInterfaceBase, ObjectDetector):
    _is_model_head = True

    default_config = {
        "patch_size": 14,  # default patch size for dinov2
        "processing_height": 480,
        "processing_width": 640,
        "use_upsampler": False,  # whether to use the upsampler model, currently not feasible
        "min_size": 480,
        "max_size": 640,
        "box_nms_thresh": 0.5,
        "img_width_dino": 518,
        "img_height_dino": 518,
        "conf_thres": 0.5,
        "num_classes": 3,
    }

    size_specific_configs = {
        "vits": {
            "embedding_dim": 384,
            "out_channels": 64,
        }
    }

    def __init__(self, encoder_size: str, **kwargs):
        # Update the default config with the size specific config and the kwargs (highest priority)
        self.cfg = self.default_config.copy()
        self.cfg.update(self.size_specific_configs[encoder_size])
        self.cfg.update(kwargs)

        n_patches = (
            self.cfg["img_height_dino"] // self.cfg["patch_size"] * self.cfg["img_width_dino"] // self.cfg["patch_size"]
        )
        batch_size = 1

        self._input_signature = {
            FM_OUTPUT_FEATURES: (batch_size, n_patches, self.cfg["embedding_dim"]),
        }
        self._output_signature = {
            MH_OBJECT_DETECTION_LABELS: (
                batch_size,
                None,
            ),
            MH_OBJECT_DETECTION_SCORES: (
                batch_size,
                None,
            ),
            MH_OBJECT_DETECTION_BOXES_NORMALIZED: (batch_size, None, 4),
        }
        super().__init__(**self.cfg)

    def annotate_output(self, x: Any) -> dict[str, torch.Tensor]:
        return {
            MH_OBJECT_DETECTION_LABELS: x["labels"],
            MH_OBJECT_DETECTION_SCORES: x["scores"],
            MH_OBJECT_DETECTION_BOXES_NORMALIZED: x["boxes"],
        }

    def deannotate_input(self, x: dict[str, torch.Tensor]) -> Any:
        return x[FM_OUTPUT_FEATURES]

    def forward(self, x: Any) -> Any:
        return ObjectDetector.forward(self, x)

    @staticmethod
    def visualize_output(
        output: dict[str, torch.Tensor],
        original_image: torch.Tensor = None,
        target_height: int = 480,
        target_width: int = 640,
    ) -> np.ndarray:
        lbl = output[MH_OBJECT_DETECTION_LABELS]
        scores = output[MH_OBJECT_DETECTION_SCORES]
        box = output[MH_OBJECT_DETECTION_BOXES_NORMALIZED]

        first_false = torch.where(scores < ObjectDetectionHead.default_config["conf_thres"])[0]
        if len(first_false) == 0:
            first_false = len(scores)

        lbl = lbl[:first_false]
        scores = scores[:first_false]
        box = box[:first_false]  # normalized to [0, 1]

        if original_image is not None:
            x = original_image
            if any([x.shape[0] != i and x.shape[2] == i for i in [1, 3]]):
                x = x.permute(2, 0, 1)
        else:
            shape = (3, target_height, target_width)
            x = torch.zeros(shape).to(torch.uint8)

        # scale boxes to the target size
        box *= torch.tensor([x.shape[2], x.shape[1], x.shape[2], x.shape[1]], device=box.device)

        pred_labels = [
            ObjectDetector.label_to_cat(label.item()) + f": {score:.3f}" for label, score in zip(lbl, scores)
        ]
        output_image = draw_bounding_boxes(x, box, pred_labels, colors="red", width=5, font_size=25)
        annotated_image = output_image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        return annotated_image

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
    path = "$ROS_WORKSPACE"/nn_engine/tests/resources/object_detection/inputs/20230710_103312.png"
    image = cv2.imread(path)
    resized = cv2.resize(image, (1920, 1080))
    im0 = torch.tensor(resized).cuda()

    # save
    save_path = "$ROS_WORKSPACE"/nn_engine/models/checkpoints/object_detection_head_vits.pth"
    detector = ObjectDetectionHead("vits")
    detector.load_state_dict(torch.load(save_path))
    detector.cuda()
    detector.eval()
    detector.half()

    preprocessing = DINOV2PreprocessingTorch(torch.float16, 518, 518)
    fm = DinoFoundationModel("vits", ignore_xformers=True, apply_final_norm=False, reshape_to_patches=False)
    fm.load_state_dict(torch.load("$ROS_WORKSPACE"/nn_engine/models/checkpoints/dinov2_vits14_pretrain.pth"))
    fm.cuda()
    fm.half()
    fm.eval()
    preprocessed = preprocessing({"preprocessing_input": im0})
    feats = fm.forward_annotated(preprocessed)
    out = detector.forward_annotated(feats)

    annotated_image = detector.visualize_output(out, im0)

    compiled = torch.compile(detector)
    compiled_out = compiled(compiled.deannotate_input(feats))
    print(compiled_out)
    print("Model compiled successfully")
