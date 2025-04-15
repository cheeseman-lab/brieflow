"""Utility functions for extracting features from image data."""

# Basic features added to all feature extractions
features_basic = {
    "area": lambda r: r.area,
    "i": lambda r: r.centroid[0],
    "j": lambda r: r.centroid[1],
    "label": lambda r: r.label,
    "bounds": lambda r: r.bbox,
}


def extract_features(data, labels, wildcards, features=None, multichannel=False):
    """Extract features from the provided image data within labeled segmentation masks.

    Args:
        data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
        labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
        wildcards (dict): Metadata to include in the output table, e.g., well, tile, etc.
        features (dict or None): Features to extract and their defining functions. Default is None.
        multichannel (bool): Flag indicating whether the data has multiple channels.

    Returns:
        pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
    """
    features = features.copy() if features else dict()
    features.update(features_basic)

    # Choose appropriate feature table based on multichannel flag
    if multichannel:
        from lib.shared.feature_table_utils import (
            feature_table_multichannel as feature_table,
        )
    else:
        from lib.shared.feature_table_utils import feature_table

    # Extract features using the feature table function
    df = feature_table(data, labels, features)

    # Add wildcard metadata to the DataFrame
    for k, v in sorted(wildcards.items()):
        df[k] = v

    return df


def extract_features_bare(
    data, labels, features=None, wildcards=None, multichannel=False
):
    """Extract features in dictionary and combine with generic region features.

    Args:
        data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
        labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict or None): Features to extract and their defining functions. Default is None.
        wildcards (dict or None): Metadata to include in the output table, e.g., well, tile, etc. Default is None.
        multichannel (bool): Flag indicating whether the data has multiple channels.

    Returns:
        pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
    """
    features = features.copy() if features else dict()
    features.update({"label": lambda r: r.label})

    # Choose appropriate feature table based on multichannel flag
    if multichannel:
        from lib.shared.feature_table_utils import (
            feature_table_multichannel as feature_table,
        )
    else:
        from lib.shared.feature_table_utils import feature_table

    # Extract features using the feature table function
    df = feature_table(data, labels, features)

    # Add wildcard metadata to the DataFrame if provided
    if wildcards is not None:
        for k, v in sorted(wildcards.items()):
            df[k] = v

    return df
