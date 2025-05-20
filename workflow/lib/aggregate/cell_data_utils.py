"""Utility functions for handling cell data in the brieflow aggregation pipeline.

This module provides helper functions for manipulating cell data, including
loading metadata columns, splitting cell data into metadata and features,
and filtering features based on channel combinations.
"""

import pandas as pd

DEFAULT_METADATA_COLS = [
    "plate",
    "well",
    "tile",
    "cell_0",
    "i_0",
    "j_0",
    "site",
    "cell_1",
    "i_1",
    "j_1",
    "distance",
    "fov_distance_0",
    "fov_distance_1",
    "sgRNA_0",
    "gene_symbol_0",
    "mapped_single_gene",
    "channels_min",
    "nucleus_i",
    "nucleus_j",
    "nucleus_bounds_0",
    "nucleus_bounds_1",
    "nucleus_bounds_2",
    "nucleus_bounds_3",
    "cell_i",
    "cell_j",
    "cell_bounds_0",
    "cell_bounds_1",
    "cell_bounds_2",
    "cell_bounds_3",
    "cytoplasm_i",
    "cytoplasm_j",
    "cytoplasm_bounds_0",
    "cytoplasm_bounds_1",
    "cytoplasm_bounds_2",
    "cytoplasm_bounds_3",
]


def load_metadata_cols(metadata_cols_fp, include_classification_cols=False):
    """Load metadata column names from a file.

    Args:
        metadata_cols_fp (str): File path to the metadata columns list.
        include_classification_cols (bool, optional): Whether to include
            classification columns. Defaults to False.

    Returns:
        list: List of metadata column names.
    """
    metadata_cols = pd.read_csv(metadata_cols_fp, header=None, sep="\t")[0].tolist()

    if include_classification_cols:
        metadata_cols += [
            "class",
            "confidence",
        ]

    return metadata_cols


def split_cell_data(cell_data, metadata_cols):
    """Splits the cell data into metadata and features.

    Args:
        cell_data (pd.DataFrame): Input DataFrame containing cell data.
        metadata_cols (list): List of column names that represent metadata.

    Returns:
        tuple: (metadata, features) where metadata is a DataFrame containing
            only metadata columns and features is a DataFrame containing all
            non-metadata columns.
    """
    # Ensure all metadata columns exist in the data
    existing_metadata_cols = [col for col in metadata_cols if col in cell_data.columns]

    # Get metadata columns
    metadata = cell_data[existing_metadata_cols].copy()

    # Get feature columns (all columns not in metadata)
    features = cell_data.drop(columns=existing_metadata_cols).copy()

    return metadata, features


def channel_combo_subset(features, channel_combo, all_channels):
    """Filter features to include only columns from specified channel combination.

    Args:
        features (pd.DataFrame): DataFrame containing feature data.
        channel_combo (list): List of channels to include.
        all_channels (list): List of all available channels.

    Returns:
        pd.DataFrame: DataFrame with features filtered to include only
            columns from the specified channel combination.
    """
    # Find channels to remove (those not in channel_combo)
    channels_to_remove = [ch for ch in all_channels if ch not in channel_combo]

    # Get all column names
    columns = features.columns.tolist()

    # Find columns to remove (those containing removed channel names)
    columns_to_remove = [
        col for col in columns if any(ch in col for ch in channels_to_remove)
    ]

    # Keep all columns except those from removed channels
    columns_to_keep = [col for col in columns if col not in columns_to_remove]

    return features[columns_to_keep]


def get_feature_table_cols(feature_cols):
    """Filter feature columns based on specific tags and compartments.

    Args:
        feature_cols (list): List of feature column names.

    Returns:
        list: Filtered list of feature column names.
    """
    # Define the specific tags to look for
    intensity_tags = [
        "mean",
        "int",
        "mass_displacement",
        "mean_edge",
        "std_edge",
        "mean_frac_0",
        "mean_frac_3",
    ]
    shape_tags = ["area", "solidity", "form_factor", "eccentricity"]
    overlap_tags = ["manders"]

    # Define the specific compartments to look for
    compartments = ["nucleus", "cell"]

    # Initialize lists to store columns for each feature type
    intensity_cols = []
    shape_cols = []
    overlap_cols = []

    # Filter columns based on compartments and tags
    for col in feature_cols:
        # Only include columns for nucleus or cell compartments
        if any(compartment in col for compartment in compartments):
            # Intensity features - must be at END of string
            if any(col.lower().endswith(tag) for tag in intensity_tags):
                intensity_cols.append(col)

            # Shape features - must be at END of string
            elif any(col.lower().endswith(tag) for tag in shape_tags):
                shape_cols.append(col)

            # Overlap features - can be anywhere in string
            elif any(tag in col.lower() for tag in overlap_tags):
                overlap_cols.append(col)

    # Create a new DataFrame with selected columns, preserving the label column if it exists
    selected_columns = []
    if "label" in feature_cols:
        selected_columns.append("label")

    # Add columns in an organized way with clear section breaks
    selected_columns.extend(intensity_cols)
    selected_columns.extend(shape_cols)
    selected_columns.extend(overlap_cols)

    return selected_columns
