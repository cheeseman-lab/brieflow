"""Core functionality for CellProfiler feature extraction using cp_measure package."""

from tifffile import imread
from itertools import (
    permutations,
)  # used to generate pairs of channels for colocalization

import numpy as np
import pandas as pd
from cp_measure.bulk import (
    get_core_measurements,
    get_correlation_measurements,
    get_multimask_measurements,
)

# contains core measurements like, intensity, shape, texture etc.
from cp_measure.core import (
    measurecolocalization,
)  # contains colocalization, neighbor, overlap measurements -- more specialized than the bulk ones.
from cp_measure.multimask import (
    measureobjectneighbors,
    measureobjectoverlap,
)  # contains colocalization, neighbor, overlap measurements -- more specialized than the bulk ones.


def get_area_features(mask, prefix=""):
    """Extract area features (subset of "Type 1" cp measure features).

    Args:
        mask (np.ndarray): Segmentation mask
        prefix (str): Prefix for feature names

    Returns:
        dict: Dictionary of extracted features of the form {'label':[measurements,],}
    """
    results = {}  # initializing dict of extracted features
    measurements = get_core_measurements().items()
    # Extract all core measurements
    try:
        for name, measure_func in measurements:
            if name in {"get_zernike", "get_ferret"}:  # only single mask features
                features = measure_func(mask)
                # Add prefix to feature names
                if prefix:
                    features = {f"{prefix}_{k}": v for k, v in features.items()}
                results.update(features)
    except Exception as e:
        print(f"Warning: Error calling {name} on {mask}: {str(e)}")


# low level feature extraction - uses cp_measure.bulk.get_core_measurements() to get all basic measurements like intensity, texture, shape, etc.
def get_single_object_features(image, mask, prefix=""):
    """Extract single-object features (Type 1: 1 image + 1 mask), excluding area features.

    Args:
        image (np.ndarray): Single channel image
        mask (np.ndarray): Segmentation mask
        prefix (str): Prefix for feature names

    Returns:
        dict: Dictionary of extracted features of the form {'label':[measurements,],}
    """
    results = {}  # initializing dict of extracted features
    measurements = get_core_measurements().items()
    # Extract all core measurements
    try:
        # Extract all core measurements except area features (handled seperately)
        for name, measure_func in measurements:
            if name not in {"get_zernike", "get_ferret"}:
                features = measure_func(mask, image)
                # Add prefix to feature names
                if prefix:
                    features = {f"{prefix}_{k}": v for k, v in features.items()}
                results.update(features)
    except Exception as e:
        print(f"Warning: Error calling {name} with {mask} and {image}: {str(e)}")

    return results


# low level feature extraction - extracts all colocalization metrics between two channels within a mask.
def get_colocalization_features(image1, image2, mask, prefix=""):
    """Extract colocalization features (Type 2: 2 images + 1 mask).

    Args:
        image1 (np.ndarray): First channel image
        image2 (np.ndarray): Second channel image
        mask (np.ndarray): Segmentation mask
        prefix (str): Prefix for feature names

    Returns:
        dict: Dictionary of extracted features of the form {'label':[measurements,],}
    """
    results = {}  # initializing dict of extracted features
    measurements = get_correlation_measurements().items()
    # Extract all correlation measurements
    try:
        # Extract all correlation measurements
        for name, measure_func in measurements:
            features = measure_func(image1, image2, mask)
            if prefix:
                features = {f"{prefix}_{k}": v for k, v in features.items()}
            results.update(features)
    except Exception as e:
        print(
            f"Warning: Error calling colocalization measurement {name} with {image1}, {image2}, and {mask}: {str(e)}"
        )

    return results


# low level feature extraction - extracts all neighbor relationship metrics between two masks within a mask.
def get_neighbor_features(mask1, mask2, prefix=""):
    """Extract neighbor relationship features (Type 3: 2 masks).

    Args:
        mask1 (np.ndarray): First segmentation mask
        mask2 (np.ndarray): Second segmentation mask
        prefix (str): Prefix for feature names

    Returns:
        dict: Dictionary of extracted features of the form {'label':[measurements,],}
    """
    results = {}  # initializing dict of extracted features
    measurements = get_multimask_measurements().items()
    try:
        # Extract all correlation measurements
        for name, measure_func in measurements:
            features = measure_func(
                mask1,
                mask2,
            )
            # Add prefix to feature names to ID source
            if prefix:
                features = {f"{prefix}_{k}": v for k, v in features.items()}
            results.update(features)
    except Exception as e:
        print(
            f"Error calling neighbor measurment {name} with {mask1} and {mask2}: {str(e)}"
        )
    return results


# INTEGRATION: orchestrates all measurements, matches the existing implementation, catering to Snakemake
# 1 Area measurements, params: mask only
# 2 Single-object features for each channel-mask combination, params: mask + 1 img
# 3 Colocalization features between all channel pairs, params: mask + 2 imgs
# 4 Neighbor features between different mask types, params: 2 masks
# NOTE: 1 & 2 both specified to be "type 1" functions on cp measure readme (fns are categorized by their parameters)
def extract_phenotype_cp_measure(
    data_phenotype,
    nuclei,
    cells,
    cytoplasms=None,
    channel_names=None,
):
    """Extract comprehensive phenotype features using cp_measure.

    Args:
        data_phenotype (np.ndarray): Multi-channel image data (CHANNELS, HEIGHT, WIDTH)
        nuclei (np.ndarray): Nuclear segmentation mask
        cells (np.ndarray): Cell segmentation mask
        cytoplasms (np.ndarray, optional): Cytoplasm segmentation mask
        channel_names (list, optional): List of channel names
        wildcards (dict, optional): Snakemake wildcards

    Returns:
        pd.DataFrame: DataFrame containing all extracted features
    """
    # Input validation
    if not isinstance(data_phenotype, np.ndarray) or data_phenotype.ndim != 3:
        raise ValueError("data_phenotype must be 3D array (channels, height, width)")
    for mask, mask_name in [
        (nuclei, "nuclei"),
        (cells, "cells"),
        (cytoplasms, "cytoplasms"),
    ]:
        if mask is not None and not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ValueError(f"{mask_name} must be a 2D array (height, width)")

    print("Starting feature extraction...")

    # Print input shapes and types
    print("\nInput data properties:")
    print(f"data_phenotype: shape={data_phenotype.shape}, dtype={data_phenotype.dtype}")
    print(f"nuclei: shape={nuclei.shape}, dtype={nuclei.dtype}")
    print(f"cells: shape={cells.shape}, dtype={cells.dtype}")
    print(f"cytoplasms: shape={cytoplasms.shape}, dtype={cytoplasms.dtype}")

    # Creating default channel names in the case that channel name list not provided
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(data_phenotype.shape[0])]

    all_features = []

    try:
        # 1. Area features (single-mask input)
        print("Computing area features...")
        MASKS = [nuclei, cells, cytoplasms]
        for mask in MASKS:
            if mask is not None and np.any(mask > 0):
                features = get_area_features(mask, f"{mask_name}__")
                if features:
                    all_features.append(pd.DataFrame(features))

        # 2. Single object features (not including area measurements, excluded in helper)
        print("Computing single object features...")
        for channel_idx, channel_name in enumerate(channel_names):
            channel_data = data_phenotype[channel_idx]

            # Process each mask type if it contains objects
            for mask, mask_name in [
                (nuclei, "nucleus"),
                (cells, "cell"),
                (cytoplasms, "cytoplasm"),
            ]:
                if mask is not None and np.any(mask > 0):
                    features = get_single_object_features(
                        channel_data, mask, f"{mask_name}_{channel_name}__"
                    )
                    if features:
                        all_features.append(pd.DataFrame(features))

        # 3. Colocalization features
        print("Computing colocalization features...")
        for (ch1_idx, ch1_name), (ch2_idx, ch2_name) in permutations(
            enumerate(channel_names), 2
        ):  # FIX: using permutations instead of combinations
            ch1_data = data_phenotype[ch1_idx]
            ch2_data = data_phenotype[ch2_idx]

            for mask, mask_name in [
                (nuclei, "nucleus"),
                (cells, "cell"),
                (cytoplasms, "cytoplasm"),
            ]:
                if mask is not None and np.any(mask > 0):
                    features = get_colocalization_features(
                        ch1_data,
                        ch2_data,
                        mask,
                        f"{mask_name}_{ch1_name}_{ch2_name}_coloc__",
                    )
                    if features:
                        all_features.append(pd.DataFrame(features))

        # 4. Neighbor features
        print("Computing neighbor features...")
        # Process each mask pair if both contain objects
        mask_pairs = [
            (nuclei, cells, "nucleus_cell"),
            (nuclei, cytoplasms, "nucleus_cytoplasm"),
            (cells, cytoplasms, "cell_cytoplasm"),
        ]

        for mask1, mask2, prefix in mask_pairs:
            if (
                mask1 is not None
                and mask2 is not None
                and np.any(mask1 > 0)
                and np.any(mask2 > 0)
            ):
                features = get_neighbor_features(mask1, mask2, f"{prefix}_neighbor__")
                if features:
                    all_features.append(pd.DataFrame(features))

    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        return pd.DataFrame()

    # Combine all features
    print("Combining features...")
    if not all_features:
        print("Warning: No features were extracted")
        return pd.DataFrame()

    features_df = pd.concat(all_features, axis=1)

    return features_df
