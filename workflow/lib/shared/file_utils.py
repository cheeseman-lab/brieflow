"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# Mapping of metadata keys to filename prefixes and data types
FILENAME_METADATA_MAPPING = {
    "plate": ["P-", str],
    "well": ["W-", str],
    "tile": ["T-", int],
    "cycle": ["C-", int],
    "gene": ["G-", str],
    "sgrna": ["SG-", str],
    "channel": ["CH-", str],
    "dataset": ["DT-", str],
}


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a structured filename based on data location, information type, and file type.

    Args:
        data_location (dict): Dictionary containing location info like well, tile, and cycle.
        info_type (str): Type of information (e.g., 'cell_features', 'sbs_reads').
        file_type (str): File extension/type (e.g., 'tsv', 'parquet', 'tiff').

    Returns:
        str: Structured filename.
    """
    parts = []

    for metadata_key, metadata_value in data_location.items():
        if metadata_key in FILENAME_METADATA_MAPPING:
            prefix, _ = FILENAME_METADATA_MAPPING[metadata_key]
            parts.append(f"{prefix}{metadata_value}")
        else:
            print(f"Unknown metadata key: {metadata_key}")

    prefix = "_".join(parts)
    filename = (
        f"{prefix}__{info_type}.{file_type}" if prefix else f"{info_type}.{file_type}"
    )

    return filename


def parse_filename(file_path: str) -> tuple:
    """Parse a structured filename from a file path to extract data location, information type, and file type.

    Args:
        file_path (str): Full file path or filename, e.g., '/path/to/W_A1_T02_C03__cell_features.tsv'.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Dictionary with keys like 'well', 'tile', 'cycle' as applicable.
            - info_type (str): The type of information (e.g., 'cell_features').
            - file_type (str): The file extension/type (e.g., 'tsv').
    """
    # Convert the input to a Path object
    path = Path(file_path)

    # Extract the filename and file extension
    filename = path.stem
    file_type = path.suffix.lstrip(".")

    # Split the filename into main parts
    parts = filename.split("__")

    # Initialize metadata dictionary and info_type variable
    metadata = {}
    info_type = None

    # Parse data location part
    if len(parts) == 2:
        location_part, info_type = parts
        elements = location_part.split("_")

        for element in elements:
            for key, (prefix, data_type) in FILENAME_METADATA_MAPPING.items():
                if element.startswith(prefix):
                    # Extract and convert the value based on the data type
                    value = element[len(prefix) :]
                    metadata[key] = data_type(value)
                    break  # Stop checking other prefixes for this element
    else:
        # If no location part, the first part is the info_type
        info_type = parts[0]

    return metadata, info_type, file_type


def validate_dtypes(df):
    """Convert DataFrame columns to the most specific data type possible with the following rules.

    - Convert strings to int if possible, or float if necessary
    - Convert floats to int if possible

    Args:
    df : pandas.DataFrame
        The DataFrame to optimize

    Returns:
    pandas.DataFrame
        A new DataFrame with optimized dtypes
    """
    for col in df.columns:
        # Skip columns that are already int64
        if pd.api.types.is_integer_dtype(df[col]):
            continue

        # Attempt to convert strings to float
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = df[col].astype("Float64")
            except ValueError:
                pass

        # Convert float to int if possible
        if pd.api.types.is_float_dtype(df[col]):
            col_subset = (
                df[col]
                .dropna()
                .sample(min(10000, df[col].notna().sum()), random_state=42)
            )
            if np.array_equal(col_subset, col_subset.astype(int)):
                df[col] = df[col].astype("Int64")

    return df
