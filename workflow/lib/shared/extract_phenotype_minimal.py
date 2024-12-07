"""Utilities for extracting minimal phenotype features from nuclei data."""

import pandas as pd

from lib.shared.feature_extraction import extract_features


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
