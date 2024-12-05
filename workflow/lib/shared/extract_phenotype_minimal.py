"""Utilities for extracting minimal phenotype features from nuclei data."""

from lib.shared.features import features_basic

import pandas as pd


def extract_phenotype_minimal(phenotype_data, nuclei_data, wildcards):
    """Extracts minimal phenotype features from the provided phenotype data.

    Args:
        phenotype_data (pandas DataFrame): DataFrame containing phenotype data.
        nuclei_data (numpy array): Array containing nuclei information.
        wildcards (dict): Metadata to include in output table.

    Returns:
        pandas DataFrame: Extracted minimal phenotype features with cell labels.
    """
    # Call _extract_features method to extract features using provided phenotype data and nuclei information
    phentoype_minimal = extract_features(
        phenotype_data, nuclei_data, wildcards, dict()
    ).rename(columns={"label": "cell"})

    if phentoype_minimal.empty:
        columns = ["area", "i", "j", "cell", "bounds", "tile", "well"]
        return pd.DataFrame(columns=columns)
    else:
        return phentoype_minimal


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
