"""Utility functions for working with feature tables."""

from collections import defaultdict
from collections.abc import Iterable

import pandas as pd
import numpy as np
import skimage.measure


def feature_table(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.

    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = regionprops(labels, intensity_image=data)

    # Initialize a defaultdict to store feature values
    results = defaultdict(list)

    # Loop through each region and compute features
    for region in regions:
        for feature, func in features.items():
            # Apply the feature function to the region and append the result to the corresponding feature list
            results[feature].append(fix_uint16(func(region)))

    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = fix_uint16(func(data, labels))

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


def feature_table_multichannel(data, labels, features, global_features=None):
    """Apply functions in feature dictionary to regions in data specified by integer labels.

    If provided, the global feature dictionary is applied to the full input data and labels.
    Results are combined in a dataframe with one row per label and one column per feature.

    Args:
        data (np.ndarray): Image data.
        labels (np.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict): Dictionary of feature names and their corresponding functions.
        global_features (dict, optional): Dictionary of global feature names and their corresponding functions.

    Returns:
        pd.DataFrame: DataFrame containing extracted features with one row per label and one column per feature.
    """
    # Extract regions from the labeled segmentation mask
    regions = regionprops_multichannel(labels, intensity_image=data)

    # Initialize a defaultdict to store feature values
    results = defaultdict(list)

    # Loop through each feature and compute features for each region
    for feature, func in features.items():
        # Check if the result of applying the function to the first region is iterable
        result_0 = func(regions[0])
        if isinstance(result_0, Iterable):
            if len(result_0) == 1:
                # If the result is a single value, apply the function to each region and append the result to the corresponding feature list
                results[feature] = [func(region)[0] for region in regions]
            else:
                # If the result is a sequence, apply the function to each region and append each element of the result to the corresponding feature list
                for result in map(func, regions):
                    for index, value in enumerate(result):
                        results[f"{feature}_{index}"].append(value)
        else:
            # If the result is not iterable, apply the function to each region and append the result to the corresponding feature list
            results[feature] = list(map(func, regions))

    # If global features are provided, compute them and add them to the results
    if global_features:
        for feature, func in global_features.items():
            # Apply the global feature function to the full input data and labels
            results[feature] = func(data, labels)

    # Convert the results dictionary to a DataFrame
    return pd.DataFrame(results)


def regionprops(labeled, intensity_image):
    """Supplement skimage.measure.regionprops with additional field `intensity_image_full` containing multi-dimensional intensity image.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Intensity image.

    Returns:
        list: List of region properties objects.
    """
    # If intensity image has more than 2 dimensions, consider only the first channel
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = intensity_image[..., 0, :, :]

    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    # Iterate over regions and add the 'intensity_image_full' attribute
    for region in regions:
        b = region.bbox  # Get bounding box coordinates
        # Extract the corresponding sub-image from the intensity image and assign it to the 'intensity_image_full' attribute
        region.intensity_image_full = intensity_image[..., b[0] : b[2], b[1] : b[3]]

    return regions


def regionprops_multichannel(labeled, intensity_image):
    """Format intensity image axes for compatibility with updated skimage.measure.regionprops that allows multichannel images.

    Some operations are faster than regionprops, others are slower.

    Args:
        labeled (np.ndarray): Labeled segmentation mask defining objects.
        intensity_image (np.ndarray): Multichannel intensity image.

    Returns:
        list: List of region properties objects.
    """
    import skimage.measure

    # If intensity image has only 2 dimensions, consider it as a single-channel image
    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        # Move the channel axis to the last position for compatibility with skimage.measure.regionprops
        base_image = np.moveaxis(
            intensity_image,
            range(intensity_image.ndim - 2),
            range(-1, -(intensity_image.ndim - 1), -1),
        )

    # Compute region properties using skimage.measure.regionprops
    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    return regions


def fix_uint16(x):
    """Pandas bug converts np.uint16 to np.int16!!!

    Args:
        x (Union[np.uint16, int]): Value to fix.

    Returns:
        Union[int, np.uint16]: Fixed value.
    """
    if isinstance(x, np.uint16):
        return int(x)
    return x
