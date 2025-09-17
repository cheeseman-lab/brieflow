"""Utility functions for handling cell data in the brieflow aggregation pipeline.

This module provides helper functions for manipulating cell data, including
loading metadata columns, splitting cell data into metadata and features,
and filtering features based on channel combinations.
"""

import pandas as pd

from lib.phenotype.constants import DEFAULT_METADATA_COLS


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


def compartment_subset(features, compartment_combos):
    """Filter features to include only columns from specified compartments.

    Args:
        features (pd.DataFrame): DataFrame containing feature data.
        compartment_combos (list): List of compartment prefixes to include
            (e.g., ['nucleus', 'cell']).

    Returns:
        pd.DataFrame: DataFrame with features filtered to include only
            columns from the specified compartments.
    """
    # Get all column names
    columns = features.columns.tolist()

    # Find columns to keep (those starting with included compartment prefixes)
    columns_to_keep = []
    for col in columns:
        for comp in compartment_combos:
            if col.startswith(comp + "_"):
                columns_to_keep.append(col)
                break

    return features[columns_to_keep]


def get_feature_table_cols(feature_cols, compartment_combos=None):
    """Filter feature columns based on specific tags and compartments.

    Args:
        feature_cols (list): List of feature column names.
        compartment_combos (list, optional): List of compartment prefixes to include
            (e.g., ['nucleus', 'cell', 'vacuole']). Defaults to ['nucleus', 'cell'].

    Returns:
        list: Filtered list of feature column names.
    """
    # Default to nucleus and cell if no compartments specified
    if compartment_combos is None:
        compartment_combos = ["nucleus", "cell"]
    
    # Define compartment-specific tags
    compartment_tags = {
        "nucleus": {
            "intensity": ["mean", "int", "mass_displacement", "mean_edge", "std_edge", "mean_frac_0", "mean_frac_3"],
            "shape": ["area", "solidity", "form_factor", "eccentricity"],
            "overlap": ["manders"]
        },
        "cell": {
            "intensity": ["mean", "int", "mass_displacement", "mean_edge", "std_edge", "mean_frac_0", "mean_frac_3"],
            "shape": ["area", "solidity", "form_factor", "eccentricity"],
            "overlap": ["manders"]
        },
        "vacuole": {
            "intensity": ["mean"],
            "shape": ["area", "diameter"],
            "distance": ["distance_to_nucleus"]
        }
    }

    # Initialize lists to store columns for each feature type
    intensity_cols = []
    shape_cols = []
    overlap_cols = []
    distance_cols = []  # New category for vacuole distance measurements

    # Filter columns based on compartments and tags
    for col in feature_cols:
        # Check if column belongs to any of the specified compartments
        for compartment in compartment_combos:
            if col.startswith(f"{compartment}_"):
                # Get tags for this compartment
                tags = compartment_tags.get(compartment, {})
                
                # Check intensity features - must be at END of string
                if any(col.lower().endswith(tag) for tag in tags.get("intensity", [])):
                    intensity_cols.append(col)
                    break
                
                # Check shape features - must be at END of string
                elif any(col.lower().endswith(tag) for tag in tags.get("shape", [])):
                    shape_cols.append(col)
                    break
                
                # Check overlap features - can be anywhere in string
                elif any(tag in col.lower() for tag in tags.get("overlap", [])):
                    overlap_cols.append(col)
                    break
                
                # Check distance features - can be anywhere in string
                elif any(tag in col.lower() for tag in tags.get("distance", [])):
                    distance_cols.append(col)
                    break

    # Create a new list with selected columns, preserving the label column if it exists
    selected_columns = []
    if "label" in feature_cols:
        selected_columns.append("label")

    # Add columns in an organized way with clear section breaks
    selected_columns.extend(intensity_cols)
    selected_columns.extend(shape_cols)
    selected_columns.extend(overlap_cols)
    selected_columns.extend(distance_cols)

    return selected_columns