"""
Cellpose Fine-Tuning Utilities

Functions for loading training data, augmentation, training, evaluation,
and visualization of Cellpose models for secondary object segmentation.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from tifffile import imread
from cellpose import models, train, io
from skimage.transform import rotate
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Reuse utilities from segment_cellpose (single source of truth)
from lib.shared.segment_cellpose import prepare_cellpose, create_cellpose_model


def load_training_data(
    image_paths: List[Union[str, Path]],
    mask_paths: List[Union[str, Path]],
    mode: str = "secondary_obj",
    channel_index: Optional[int] = None,
    dapi_index: Optional[int] = None,
    cyto_index: Optional[int] = None,
    helper_index: Optional[int] = None,
    logscale: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load paired images and masks for Cellpose training.

    Supports different preprocessing modes to match deployment functions:
    - "secondary_obj": For segment_second_objs_ml (single channel, log scaling)
    - "cells": For segment_cellpose with cells=True (3-channel RGB)
    - "nuclei": For segment_cellpose with cells=False (DAPI only)

    Parameters
    ----------
    image_paths : List[str | Path]
        Paths to image files (TIFF format, can be multi-channel).
    mask_paths : List[str | Path]
        Paths to mask files (NPY format, labeled masks where each object
        has a unique integer ID and background is 0).
    mode : str
        Preprocessing mode. Options:
        - "secondary_obj": For segment_second_objs_ml (requires channel_index)
        - "cells": For segment_cellpose cells (requires dapi_index, cyto_index)
        - "nuclei": For segment_cellpose nuclei only (requires dapi_index)
    channel_index : int, optional
        Channel index for secondary_obj mode.
    dapi_index : int, optional
        DAPI channel index for cells/nuclei modes.
    cyto_index : int, optional
        Cytoplasm channel index for cells mode.
    helper_index : int, optional
        Helper channel index for cells mode (optional).
    logscale : bool
        Apply log scaling preprocessing. Default True.

    Returns
    -------
    images : List[np.ndarray]
        List of preprocessed image arrays (uint8).
        - secondary_obj/nuclei: 2D arrays [height, width]
        - cells: 3D arrays [3, height, width]
    masks : List[np.ndarray]
        List of 2D labeled mask arrays (int32).

    Raises
    ------
    ValueError
        If number of images and masks don't match, required indices not provided,
        or if dimensions mismatch.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) must match "
            f"number of masks ({len(mask_paths)})"
        )

    # Validate mode and required parameters
    if mode == "secondary_obj":
        if channel_index is None:
            raise ValueError("channel_index is required for mode='secondary_obj'")
    elif mode == "cells":
        if dapi_index is None or cyto_index is None:
            raise ValueError(
                "dapi_index and cyto_index are required for mode='cells'"
            )
    elif mode == "nuclei":
        if dapi_index is None:
            raise ValueError("dapi_index is required for mode='nuclei'")
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Valid options: 'secondary_obj', 'cells', 'nuclei'"
        )

    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        # Load image
        img = imread(str(img_path))

        # Apply mode-specific preprocessing using prepare_cellpose (single source of truth)
        if mode == "secondary_obj":
            # Use prepare_cellpose with target channel as cyto, extract green channel
            # Green channel (index 1) has log scaling + max normalization
            rgb = prepare_cellpose(
                img,
                dapi_index=channel_index,  # Dummy - will use cyto
                cyto_index=channel_index,   # Target channel
                helper_index=None,
                logscale=logscale,
            )
            processed_img = rgb[1]  # Extract green (log scaled + normalized)

        elif mode == "cells":
            # Full RGB output from prepare_cellpose
            processed_img = prepare_cellpose(
                img,
                dapi_index=dapi_index,
                cyto_index=cyto_index,
                helper_index=helper_index,
                logscale=logscale,
            )

        elif mode == "nuclei":
            # Use prepare_cellpose and extract DAPI (blue) channel
            rgb = prepare_cellpose(
                img,
                dapi_index=dapi_index,
                cyto_index=dapi_index,  # Dummy
                helper_index=None,
                logscale=False,  # DAPI uses percentile norm, not log scale
            )
            processed_img = rgb[2]  # Extract blue (DAPI with percentile norm)

        # Load mask (support both .npy and .tif/.tiff formats)
        mask_path_str = str(mask_path)
        if mask_path_str.endswith('.npy'):
            mask = np.load(mask_path_str)
        else:
            mask = imread(mask_path_str)
        if mask.ndim > 2:
            raise ValueError(f"Mask at {mask_path} should be 2D, got {mask.ndim}D")

        # Validate dimensions match (compare 2D shapes)
        img_shape_2d = processed_img.shape[-2:] if processed_img.ndim > 2 else processed_img.shape
        if img_shape_2d != mask.shape:
            raise ValueError(
                f"Image shape {img_shape_2d} doesn't match mask shape {mask.shape} "
                f"for {img_path}"
            )

        images.append(processed_img)
        masks.append(mask.astype(np.int32))

    print(f"Loaded {len(images)} image-mask pairs (mode={mode})")
    return images, masks


def augment_training_data(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    rotations: bool = True,
    flips: bool = True,
    intensity_scaling: bool = True,
    intensity_range: Tuple[float, float] = (0.8, 1.2),
    noise: bool = False,
    noise_std: float = 0.02,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply data augmentation to expand small training datasets.

    For a dataset of N images, this can produce up to 8N augmented samples
    (4 rotations x 2 flip states).

    Parameters
    ----------
    images : List[np.ndarray]
        List of 2D image arrays.
    masks : List[np.ndarray]
        List of 2D labeled mask arrays.
    rotations : bool
        Apply 90, 180, 270 degree rotations.
    flips : bool
        Apply horizontal and vertical flips.
    intensity_scaling : bool
        Apply random intensity scaling.
    intensity_range : Tuple[float, float]
        Range for intensity scaling factor.
    noise : bool
        Add Gaussian noise.
    noise_std : float
        Standard deviation of Gaussian noise.

    Returns
    -------
    aug_images : List[np.ndarray]
        Augmented images (includes originals).
    aug_masks : List[np.ndarray]
        Augmented masks (includes originals).
    """
    aug_images = []
    aug_masks = []

    for img, mask in zip(images, masks):
        # Start with original
        variants = [(img.copy(), mask.copy())]

        # Rotations (90, 180, 270 degrees)
        if rotations:
            for k in [1, 2, 3]:  # k*90 degrees
                rot_img = np.rot90(img, k)
                rot_mask = np.rot90(mask, k)
                variants.append((rot_img.copy(), rot_mask.copy()))

        # Flips
        if flips:
            current_variants = variants.copy()
            for v_img, v_mask in current_variants:
                # Horizontal flip
                flip_img = np.fliplr(v_img)
                flip_mask = np.fliplr(v_mask)
                variants.append((flip_img.copy(), flip_mask.copy()))

        # Intensity modifications (applied to each variant)
        final_variants = []
        for v_img, v_mask in variants:
            # Original intensity
            final_variants.append((v_img.copy(), v_mask.copy()))

            # Intensity scaling
            if intensity_scaling:
                scale = random.uniform(intensity_range[0], intensity_range[1])
                scaled_img = np.clip(v_img * scale, 0, 1)
                final_variants.append((scaled_img.astype(np.float32), v_mask.copy()))

            # Noise
            if noise:
                noisy_img = v_img + np.random.normal(0, noise_std, v_img.shape)
                noisy_img = np.clip(noisy_img, 0, 1)
                final_variants.append((noisy_img.astype(np.float32), v_mask.copy()))

        for v_img, v_mask in final_variants:
            aug_images.append(v_img)
            aug_masks.append(v_mask)

    print(
        f"Augmentation: {len(images)} original → {len(aug_images)} samples "
        f"({len(aug_images) / len(images):.1f}x expansion)"
    )
    return aug_images, aug_masks


def prepare_cellpose_training(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split data into training and test sets for Cellpose.

    Parameters
    ----------
    images : List[np.ndarray]
        List of 2D image arrays.
    masks : List[np.ndarray]
        List of 2D labeled mask arrays.
    test_fraction : float
        Fraction of data to use for testing (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_images : List[np.ndarray]
    train_masks : List[np.ndarray]
    test_images : List[np.ndarray]
    test_masks : List[np.ndarray]
    """
    n_samples = len(images)
    n_test = max(1, int(n_samples * test_fraction))

    # Shuffle indices
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]

    print(f"Split: {len(train_images)} training, {len(test_images)} test samples")
    return train_images, train_masks, test_images, test_masks


def train_cellpose(
    train_images: List[np.ndarray],
    train_masks: List[np.ndarray],
    test_images: Optional[List[np.ndarray]] = None,
    test_masks: Optional[List[np.ndarray]] = None,
    base_model: str = "cpsam",
    n_epochs: int = 500,
    learning_rate: float = 0.1,
    weight_decay: float = 1e-5,
    batch_size: int = 8,
    save_path: Union[str, Path] = "models",
    model_name: str = "cpsam_secondary_obj",
    gpu: bool = True,
    channels: List[int] = None,
) -> models.CellposeModel:
    """
    Fine-tune a Cellpose model on custom training data.

    Parameters
    ----------
    train_images : List[np.ndarray]
        Training images (2D arrays).
    train_masks : List[np.ndarray]
        Training masks (labeled 2D arrays).
    test_images : List[np.ndarray], optional
        Test images for validation during training.
    test_masks : List[np.ndarray], optional
        Test masks for validation.
    base_model : str
        Base model to fine-tune from. Options: "cpsam", "cyto3", "cyto2", "nuclei".
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization weight.
    batch_size : int
        Training batch size.
    save_path : str | Path
        Directory to save trained model.
    model_name : str
        Name for the saved model.
    gpu : bool
        Use GPU acceleration if available.
    channels : List[int], optional
        Channel configuration for Cellpose. Default [0, 0] for grayscale.

    Returns
    -------
    model : CellposeModel
        Trained Cellpose model.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = [0, 0]  # Grayscale

    # Set up logging to see training progress
    io.logger_setup()

    print(f"Initializing model from base: {base_model}")
    print(f"Training parameters: epochs={n_epochs}, lr={learning_rate}, batch={batch_size}")

    # Initialize model with version-aware helper (validates model compatibility)
    model = create_cellpose_model(base_model, gpu=gpu)

    print(f"Starting training with {len(train_images)} samples...")

    # In Cellpose 3.0+, use train.train_seg() instead of model.train()
    # Note: channels is not a parameter for train_seg - images should be pre-formatted
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images,
        test_labels=test_masks,
        save_path=str(save_path),
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        model_name=model_name,
    )

    print(f"Training complete. Model saved to: {model_path}")

    # Load and return the trained model
    trained_model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    return trained_model


def load_trained_model(
    model_path: Union[str, Path],
    gpu: bool = True,
) -> models.CellposeModel:
    """
    Load a fine-tuned Cellpose model.

    Parameters
    ----------
    model_path : str | Path
        Path to saved model file.
    gpu : bool
        Use GPU acceleration.

    Returns
    -------
    model : CellposeModel
        Loaded Cellpose model ready for inference.
    """
    model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
    print(f"Loaded model from: {model_path}")
    return model


def predict_masks(
    model: models.CellposeModel,
    images: List[np.ndarray],
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    channels: List[int] = None,
) -> List[np.ndarray]:
    """
    Run inference on images using a Cellpose model.

    Parameters
    ----------
    model : CellposeModel
        Cellpose model (base or fine-tuned).
    images : List[np.ndarray]
        List of 2D images to segment.
    diameter : float, optional
        Expected object diameter. None for auto-estimation.
    flow_threshold : float
        Flow error threshold.
    cellprob_threshold : float
        Cell probability threshold.
    channels : List[int], optional
        Channel configuration. Default [0, 0] for grayscale.

    Returns
    -------
    masks : List[np.ndarray]
        Predicted segmentation masks.
    """
    if channels is None:
        channels = [0, 0]

    masks, flows, styles = model.eval(
        images,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    return masks


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union between predicted and ground truth masks.

    This computes the average IoU across all objects.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted labeled mask.
    gt_mask : np.ndarray
        Ground truth labeled mask.

    Returns
    -------
    iou : float
        Mean IoU score (0 to 1).
    """
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def calculate_object_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate object-level metrics (precision, recall, F1).

    An object is considered a true positive if it overlaps with a ground truth
    object with IoU >= threshold.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted labeled mask.
    gt_mask : np.ndarray
        Ground truth labeled mask.
    iou_threshold : float
        IoU threshold for matching objects.

    Returns
    -------
    metrics : dict
        Dictionary with 'precision', 'recall', 'f1', 'n_pred', 'n_gt', 'n_tp'.
    """
    pred_labels = np.unique(pred_mask[pred_mask > 0])
    gt_labels = np.unique(gt_mask[gt_mask > 0])

    n_pred = len(pred_labels)
    n_gt = len(gt_labels)

    if n_pred == 0 and n_gt == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "n_pred": 0,
            "n_gt": 0,
            "n_tp": 0,
        }

    if n_pred == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_pred": 0,
            "n_gt": n_gt,
            "n_tp": 0,
        }

    if n_gt == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_pred": n_pred,
            "n_gt": 0,
            "n_tp": 0,
        }

    # Match predictions to ground truth
    matched_gt = set()
    tp = 0

    for pred_label in pred_labels:
        pred_region = pred_mask == pred_label
        best_iou = 0
        best_gt = None

        for gt_label in gt_labels:
            if gt_label in matched_gt:
                continue
            gt_region = gt_mask == gt_label

            intersection = np.logical_and(pred_region, gt_region).sum()
            union = np.logical_or(pred_region, gt_region).sum()
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_label

        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)

    precision = tp / n_pred if n_pred > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_tp": tp,
    }


def evaluate_segmentation(
    model: models.CellposeModel,
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model performance on a test set.

    Parameters
    ----------
    model : CellposeModel
        Cellpose model to evaluate.
    images : List[np.ndarray]
        Test images.
    gt_masks : List[np.ndarray]
        Ground truth masks.
    diameter : float, optional
        Object diameter for inference.
    flow_threshold : float
        Flow threshold for inference.
    cellprob_threshold : float
        Cell probability threshold.
    iou_threshold : float
        IoU threshold for object matching.

    Returns
    -------
    metrics : dict
        Aggregated metrics: mean_iou, mean_precision, mean_recall, mean_f1.
    """
    pred_masks = predict_masks(
        model,
        images,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    ious = []
    precisions = []
    recalls = []
    f1s = []

    for pred, gt in zip(pred_masks, gt_masks):
        iou = calculate_iou(pred, gt)
        metrics = calculate_object_metrics(pred, gt, iou_threshold)

        ious.append(iou)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    return {
        "mean_iou": np.mean(ious),
        "mean_precision": np.mean(precisions),
        "mean_recall": np.mean(recalls),
        "mean_f1": np.mean(f1s),
        "per_image_iou": ious,
        "per_image_precision": precisions,
        "per_image_recall": recalls,
        "per_image_f1": f1s,
    }


def random_label_cmap(n_labels: int = 256, seed: int = 42) -> ListedColormap:
    """Create a random colormap for labeled masks."""
    np.random.seed(seed)
    colors = np.random.rand(n_labels, 3)
    colors[0] = [0, 0, 0]  # Background is black
    return ListedColormap(colors)


def visualize_comparison(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Visualize side-by-side comparison of prediction vs ground truth.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    pred_mask : np.ndarray
        Predicted segmentation mask.
    gt_mask : np.ndarray
        Ground truth mask.
    title : str
        Figure title.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth
    cmap = random_label_cmap(max(gt_mask.max(), pred_mask.max()) + 1)
    axes[1].imshow(gt_mask, cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"Ground Truth ({gt_mask.max()} objects)")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(pred_mask, cmap=cmap, interpolation="nearest")
    axes[2].set_title(f"Prediction ({pred_mask.max()} objects)")
    axes[2].axis("off")

    # Overlay
    overlay = np.zeros((*image.shape, 3))
    overlay[..., 0] = image  # Red channel = image
    overlay[..., 1] = (gt_mask > 0).astype(float) * 0.5  # Green = GT
    overlay[..., 2] = (pred_mask > 0).astype(float) * 0.5  # Blue = prediction
    axes[3].imshow(np.clip(overlay, 0, 1))
    axes[3].set_title("Overlay (G=GT, B=Pred)")
    axes[3].axis("off")

    # Calculate metrics
    iou = calculate_iou(pred_mask, gt_mask)
    metrics = calculate_object_metrics(pred_mask, gt_mask)

    fig.suptitle(
        f"{title}\nIoU: {iou:.3f} | Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}",
        fontsize=12,
    )

    plt.tight_layout()
    return fig


def visualize_training_sample(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Visualize a single training sample (image + mask).

    Parameters
    ----------
    image : np.ndarray
        Training image.
    mask : np.ndarray
        Corresponding mask.
    title : str
        Figure title.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    cmap = random_label_cmap(mask.max() + 1)
    axes[1].imshow(mask, cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"Mask ({mask.max()} objects)")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig
