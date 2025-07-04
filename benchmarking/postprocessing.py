import os
import json

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def rmv_image_extension(image_name: str):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        return image_name[:-4]
    return image_name


def describe_array(array: np.ndarray):
    """Return a dictionary with summary statistics of the array."""
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "25_percentile": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "75_percentile": float(np.percentile(array, 75)),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def apply_color_map_and_save(
    outputs: list, orig_h: int, orig_w: int, output_dir: str, benchmark: dict, image_files: list
):
    output_subdir = os.path.join(output_dir, "output")
    os.makedirs(output_subdir, exist_ok=True)

    for idx, depth in enumerate(tqdm(outputs, desc="Postprocessing")):
        # Save the depth output
        depth = np.array(depth).squeeze()
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"{output_subdir}/{rmv_image_extension(image_files[idx])}_depth.png", colored_depth)


def compute_relative_error(
    outputs: list, orig_h: int, orig_w: int, output_dir: str, benchmark: dict, image_files: list
):
    output_images_dir = os.path.join(output_dir, "output", "images")
    output_scores_dir = os.path.join(output_dir, "output", "scores")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_scores_dir, exist_ok=True)

    errors = []
    for idx, depth in enumerate(tqdm(outputs, desc="Postprocessing")):
        # rescale depth output to original size
        depth = np.array(depth).squeeze()
        depth = cv2.resize(depth.astype("float32"), (orig_w, orig_h))

        # Save the depth output
        colored_depth = cv2.applyColorMap(depth.astype("uint8"), cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"{output_images_dir}/{rmv_image_extension(image_files[idx])}_depth.png", colored_depth)

        # Load the numpy ground truth depth
        gt_depth = np.load(os.path.join(benchmark["ground_truth_dir"], f"{image_files[idx].replace('.jpg', '.npy')}"))

        assert depth.shape == gt_depth.shape

        # Compute the relative error
        valid_mask = gt_depth != 0
        difference = np.abs(gt_depth - depth.astype(np.float32))
        errors.append(describe_array(difference[valid_mask]))

        # save heatmap of errors
        error_plot = np.zeros_like(gt_depth)
        error_plot[valid_mask] = difference[valid_mask]

        plt.figure(figsize=(8, 2))
        plt.imshow(error_plot, cmap="hot", vmin=0, vmax=np.max(error_plot), interpolation="nearest")
        plt.colorbar(label="Absolute error in meters")

        # Add labels and a title
        plt.title("Heatmap of Errors")
        plt.xlabel("Column")
        plt.ylabel("Row")

        # Save the figure to a file
        plt.savefig(f"{output_images_dir}/{rmv_image_extension(image_files[idx])}_absolute_error.png")

        # Close the plot to avoid displaying it
        plt.close()

    # Save the relative errors in json format
    with open(f"{output_scores_dir}/relative_errors.json", "w") as f:
        json.dump(
            {
                "errors": errors,
                "avg_mean_error": np.mean([error["mean"] for error in errors]),
                "avg_std_error": np.mean([error["std"] for error in errors]),
                "avg_25_percentile_error": np.mean([error["25_percentile"] for error in errors]),
                "avg_median_error": np.mean([error["median"] for error in errors]),
                "avg_75_percentile_error": np.mean([error["75_percentile"] for error in errors]),
                "avg_max_error": np.mean([error["max"] for error in errors]),
                "avg_min_error": np.mean([error["min"] for error in errors]),
            },
            f,
        )
