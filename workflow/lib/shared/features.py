"""Functions and variables for extracting features from image regions."""

import numpy as np


def correlate_channels_masked(r, first, second):
    """Cross-correlation between non-zero pixels of two channels within a masked region.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.
        first (int): Index of the first channel.
        second (int): Index of the second channel.

    Returns:
        float: Mean cross-correlation coefficient between the non-zero pixels of the two channels.
    """
    # Extract intensity images for the specified channels from the region
    A = masked(r, first)
    B = masked(r, second)

    # Filter out zero pixels from both channels
    filt = (A > 0) & (B > 0)
    # If no non-zero pixels are found, return NaN
    if filt.sum() == 0:
        return np.nan

    # Filter the intensity values based on the non-zero pixel indices
    A = A[filt]
    B = B[filt]
    # Calculate the cross-correlation coefficient between the two channels
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    # Return the mean cross-correlation coefficient
    return corr.mean()


def masked(r, index):
    """Extract masked intensity image for a specific channel index from a region.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.
        index (int): Index of the channel to extract.

    Returns:
        array: Masked intensity image for the specified channel index.
    """
    return r.intensity_image_full[index][r.image]


def correlate_channels_multichannel(r, first, second):
    """Compute cross-correlation between non-zero pixels in the specified channels.

    This function is intended to be used with ops.utils.regionprops_multichannel.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.
        first (int): Index of the first channel.
        second (int): Index of the second channel.

    Returns:
        float: Cross-correlation between the specified channels.
    """
    # Extract intensity images for the specified channels
    A, B = r.intensity_image[..., [first, second]].T

    # Filter out zero pixels
    filt = (A > 0) & (B > 0)
    if filt.sum() == 0:
        return np.nan

    # Compute cross-correlation
    A = A[filt]
    B = B[filt]
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    return corr.mean()


def count_labels(labels, return_list=False):
    """Count the unique non-zero labels in a labeled segmentation mask.

    Args:
        labels (numpy array): Labeled segmentation mask.
        return_list (bool): Flag indicating whether to return the list of unique labels along with the count.

    Returns:
        int or tuple: Number of unique non-zero labels. If return_list is True, returns a tuple containing the count
      and the list of unique labels.
    """
    # Get unique labels in the segmentation mask
    uniques = np.unique(labels)
    # Remove the background label (0)
    ls = np.delete(uniques, np.where(uniques == 0))
    # Count the unique non-zero labels
    num_labels = len(ls)
    # Return the count or both count and list of unique labels based on return_list flag
    if return_list:
        return num_labels, ls
    return num_labels


# FEATURES
# These dictionaries define various features to be extracted from image regions represented by skimage.measure.regionprops objects.

# Intensity features
intensity = {
    "mean": lambda r: r.intensity_image[r.image].mean(),
    "median": lambda r: np.median(r.intensity_image[r.image]),
    "max": lambda r: r.intensity_image[r.image].max(),
    "min": lambda r: r.intensity_image[r.image].min(),
}

# Intensity features for multichannel images
intensity_multichannel = {
    "mean": lambda r: r.intensity_image[r.image].mean(axis=0),
    "median": lambda r: np.median(r.intensity_image[r.image], axis=0),
    "max": lambda r: r.intensity_image[r.image].max(axis=0),
    "min": lambda r: r.intensity_image[r.image].min(axis=0),
}

# Geometry features
geometry = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "bounds": lambda r: r.bbox,
    "label": lambda r: r.label,
    "eccentricity": lambda r: r.eccentricity,
    "solidity": lambda r: r.solidity,
    "convex_area": lambda r: r.convex_area,
    "perimeter": lambda r: r.perimeter,
}

# Frameshift features
frameshift = {
    "dapi_ha_corr": lambda r: correlate_channels_masked(r, 0, 1),
    "dapi_myc_corr": lambda r: correlate_channels_masked(r, 0, 2),
    "ha_median": lambda r: np.median(masked(r, 1)),
    "myc_median": lambda r: np.median(masked(r, 2)),
    "cell": lambda r: r.label,
}

# Frameshift features for multichannel images
frameshift_multichannel = {
    "dapi_ha_corr": lambda r: correlate_channels_multichannel(r, 0, 1),
    "dapi_myc_corr": lambda r: correlate_channels_multichannel(r, 0, 2),
    "ha_myc_medians": lambda r: np.median(r.intensity_image[r.image, 1:3], axis=0),
    "cell": lambda r: r.label,
}

# Translocation features
translocation = {
    "dapi_gfp_corr": lambda r: correlate_channels_masked(r, 0, 1),
    "dapi_mean": lambda r: masked(r, 0).mean(),
    "dapi_median": lambda r: np.median(masked(r, 0)),
    "gfp_median": lambda r: np.median(masked(r, 1)),
    "gfp_mean": lambda r: masked(r, 1).mean(),
    "dapi_int": lambda r: masked(r, 0).sum(),
    "gfp_int": lambda r: masked(r, 1).sum(),
    "dapi_max": lambda r: masked(r, 0).max(),
    "gfp_max": lambda r: masked(r, 1).max(),
}

# Translocation features for multichannel images
translocation_multichannel = {
    "dapi_gfp_corr": lambda r: correlate_channels_multichannel(r, 0, 1),
    "dapi_gfp_means": lambda r: r.intensity_image[r.image, :2].mean(axis=0),
    "dapi_gfp_medians": lambda r: np.median(r.intensity_image[r.image, :2], axis=0),
    "dapi_gfp_ints": lambda r: r.intensity_image[r.image, :2].sum(axis=0),
    "dapi_gfp_maxs": lambda r: r.intensity_image[r.image, :2].max(axis=0),
}

# Foci features
foci = {
    "foci_count": lambda r: count_labels(r.intensity_image),
    "foci_area": lambda r: (r.intensity_image > 0).sum(),
}

# ViewRNA features
viewRNA = {
    "cy3_median": lambda r: np.median(masked(r, 1)),
    "cy5_median": lambda r: np.median(masked(r, 2)),
    "cy5_80p": lambda r: np.percentile(masked(r, 2), 80),
    "cy3_int": lambda r: masked(r, 1).sum(),
    "cy5_int": lambda r: masked(r, 2).sum(),
    "cy5_mean": lambda r: masked(r, 2).sum(),
    "cy5_max": lambda r: masked(r, 2).max(),
}

# List of all feature dictionaries
all_features = [intensity, geometry, translocation, frameshift, viewRNA]


# This function validates the uniqueness of feature names across all feature dictionaries.
def validate_features() -> None:
    """Validate that the feature names are unique across all feature dictionaries.

    Raises:
        AssertionError: If there are duplicate feature names.
    """
    names = sum(map(list, all_features), [])
    assert len(names) == len(set(names))


# This function creates a feature dictionary containing specific feature names.
def make_feature_dict(feature_names: tuple) -> dict:
    """Create a feature dictionary containing specific feature names.

    Args:
        feature_names (tuple): Tuple of feature names to include in the dictionary.

    Returns:
        dict: Dictionary containing the specified feature names.
    """
    features = {}
    [features.update(d) for d in all_features]
    return {n: features[n] for n in feature_names}


# Validate the uniqueness of feature names
validate_features()

# Basic features including area, centroid coordinates, label, and bounding box
features_basic = make_feature_dict(("area", "i", "j", "label", "bounds"))
