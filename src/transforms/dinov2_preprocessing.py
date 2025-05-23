from src.transforms.abstract_transform import AbstractPreprocessing
from src.nn_engine.naming_convention import *
from src.nn_engine.shape_utils import assert_correct_io_shapes, is_io_compatible, assert_correct_types

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
import os
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
from typing import Tuple
from argparse import ArgumentParser
from tqdm import tqdm

########################################
### Input transformation in CV2 (LEGACY)
########################################


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method)

    sample["disparity"] = cv2.resize(sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize_CV2(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(
                    torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...],
                    (height, width),
                    mode="nearest",
                ).numpy()[0, 0]

            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


def get_transform_cv2(target_height, target_width):
    transform = Compose(
        [
            Resize_CV2(
                width=target_width,
                height=target_height,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    return transform


def process_image_cv2(image, target_height=518, target_width=518):
    orig_shape = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = get_transform_cv2(target_height, target_width)({"image": image})["image"]  # C, H, W
    image = image[None]  # B, C, H, W
    return image, orig_shape


def load_image_cv2(filepath, target_height=518, target_width=518) -> Tuple[np.ndarray, Tuple[int, int]]:
    image = cv2.imread(filepath)  # H, W, C
    image, orig_shape = process_image_cv2(image, target_height, target_width)
    return image, orig_shape


################################################################################
### Input transformation in pure PyTorch so that images can be processed on GPU
################################################################################


class BGR2RGB(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        """Image is of shape (C, H, W)"""
        return sample.flip(0)


class AddBatchDim(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample[None]


class CustomToTensor(object):
    def __init__(self, torch_output_type: torch.dtype):
        self.torch_output_type = torch_output_type

    def __call__(self, sample):
        sample = torch.permute(sample, (2, 0, 1))
        return sample.to(self.torch_output_type) / 255.0


def get_transform_torch(precision=torch.float32, target_height=518, target_width=518):
    transform = Compose(
        [
            CustomToTensor(precision),
            BGR2RGB(),
            Resize((target_height, target_width), interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddBatchDim(),
        ]
    )

    return transform


class DINOV2PreprocessingTorch(torch.nn.Module, AbstractPreprocessing):
    """Preprocess transformation for DINO V2 implemented in plain pytorch which allows are processed on GPU."""

    def __init__(
        self, fm_signature: dict[str, tuple], fm_type: torch.dtype, canonical_height: int, canonical_width: int
    ):
        super(DINOV2PreprocessingTorch, self).__init__()
        self.canonical_height = canonical_height
        self.canonical_width = canonical_width

        self.target_height = fm_signature[FM_INPUT][-2]
        self.target_width = fm_signature[FM_INPUT][-1]

        self.n_channels = 3
        self._input_type = torch.uint8  # input should always be an image hence uint8
        self._output_type = fm_type

        self.custom_to_tensor = CustomToTensor(fm_type)
        self.bgr2rgb = BGR2RGB()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.add_batch_dim = AddBatchDim()

        assert is_io_compatible(self.output_signature, fm_signature), (
            "Output signature doesn't fit the input signature of the foundation model"
        )

    @assert_correct_types
    @assert_correct_io_shapes
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.custom_to_tensor(x[PREPROCESSING_INPUT])
        x = self.bgr2rgb(x)
        x = self.add_batch_dim(x)
        x = F.interpolate(
            x,
            (self.target_height, self.target_width),
            mode="bicubic",
            align_corners=False,
        )  # align_corners=False is necessary to match opencv implementation
        x = self.normalize(x)
        return {FM_INPUT: x.contiguous()}

    @property
    def input_signature(self) -> dict[str, tuple]:
        return {
            PREPROCESSING_INPUT: (self.canonical_height, self.canonical_width, self.n_channels)
        }  # None means that it can accept any size

    @property
    def output_signature(self) -> dict[str, tuple]:
        batch_size = 1
        return {FM_INPUT: (batch_size, self.n_channels, self.target_height, self.target_width)}

    @property
    def input_type(self) -> torch.dtype:
        return self._input_type

    @property
    def output_type(self) -> torch.dtype:
        return self._output_type


########################################################
### Compare the CV2 and PyTorch transformations
### TODO: Benchmark the transformations in terms of kitti performance
########################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--compare_transforms",
        action="store_true",
        help="Compare the CV2 and PyTorch transformations. If not Preprocess class is called to allow for debugging.",
    )
    args = parser.parse_args()

    if args.compare_transforms:
        ### Compare the CV2 and PyTorch transformations

        np.random.seed(0)
        input_np = cv2.imread("$ROS_WORKSPACE"/nn_engine/resources/cheetah/frames/frame_0366.jpg")

        # First comparison
        transform1 = Compose([BGR2RGB()])
        assert (
            transform1(torch.tensor(input_np.transpose((2, 0, 1))))
            == cv2.cvtColor(input_np, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        ).all()

        # Second comparison
        transform2 = Compose(
            [
                CustomToTensor(torch.float32),
                BGR2RGB(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def transform2_old(input_np):
            image = cv2.cvtColor(input_np, cv2.COLOR_BGR2RGB) / 255.0
            _t = Compose(
                [
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )
            return _t({"image": image})["image"]

        assert np.allclose(transform2(torch.tensor(input_np)), transform2_old(input_np))

        # Final comparison
        transform3 = DINOV2PreprocessingTorch(torch.float32)
        a = transform3(torch.tensor(input_np))[0]
        b = process_image_cv2(input_np)[0]

        assert np.allclose(a, b, atol=1e-3)

        print("The transformations are equivalent.")

    else:
        ### Run the Preprocess class for debugging

        input_dir = "$ROS_WORKSPACE"/nn_engine/resources/cheetah/frames"
        loaded_images = []
        for img in tqdm(sorted(os.listdir(input_dir))[100:105], desc="Loading images"):
            loaded_images.append(
                torch.tensor(
                    cv2.imread(os.path.join(input_dir, img)),
                    device=torch.device("cuda"),
                )
            )
        pipeline = DINOV2PreprocessingTorch(torch.float16).to(device=torch.device("cuda"))
        out = pipeline(loaded_images[0])
