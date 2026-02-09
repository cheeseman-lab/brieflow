"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

from pathlib import Path

from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pandas as pd
import numpy as np

# Mapping of metadata keys to filename prefixes and data types
FILENAME_METADATA_MAPPING = {
    "plate": ["P-", str],
    "well": ["W-", str],
    "tile": ["T-", int],
    "cycle": ["C-", int],
    "round": ["R-", str],
    "cell_class": ["CeCl-", str],
    "channel_combo": ["ChCo-", str],
    "gene": ["G-", str],
    "sgrna": ["SG-", str],
    "channel": ["CH-", str],
    "leiden_resolution": ["LR-", float],
    "cluster_benchmark": ["CB-", str],
}


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a structured filename based on data location, information type, and file type.

    Produces flat filenames with metadata encoded as prefixes, e.g.:
        P-plate1_W-A1_T-01__aligned.tiff

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


def get_hcs_nested_path(
    data_location: dict,
    info_type: str,
    file_type: str = "zarr",
    subdirectory: str = None,
) -> str:
    """Generate an HCS-layout nested path for zarr stores within a plate zarr.

    Produces paths like:
        1.zarr/A/1/0/aligned.zarr
        1.zarr/A/1/0/3/image.zarr  (with cycle)
        1.zarr/A/1/0/labels/nuclei.zarr  (with subdirectory)

    The plate value gets a ``.zarr`` suffix, row and col become directory
    levels (matching HCS row/column convention), and the info_type file
    also gets a ``.zarr`` extension by default.

    Args:
        data_location (dict): Must contain 'plate', 'row', 'col', 'tile'.
            May optionally contain 'cycle'.
        info_type (str): Type of information (e.g., 'aligned', 'nuclei').
        file_type (str): File extension (default 'zarr').
        subdirectory (str): Optional subdirectory to insert before the store
            name (e.g., 'labels' for OME-NGFF label stores).

    Returns:
        str: HCS nested path string.
    """
    plate = data_location["plate"]
    parts = [
        f"{plate}.zarr",
        data_location["row"],
        data_location["col"],
        data_location["tile"],
    ]
    if "cycle" in data_location:
        parts.append(str(data_location["cycle"]))
    if subdirectory:
        parts.append(subdirectory)
    parts.append(f"{info_type}.{file_type}")
    return str(Path(*parts))


def get_nested_path(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a nested directory path with metadata encoded as directory levels.

    Produces nested paths like:
        plate1/A1/01/aligned.tiff

    The data_location keys become directory levels (in insertion order),
    and the filename is simply ``{info_type}.{file_type}``.

    Args:
        data_location (dict): Dictionary containing location info like well, tile, and cycle.
            Values become directory names in the order provided.
        info_type (str): Type of information (e.g., 'aligned', 'nuclei').
        file_type (str): File extension/type (e.g., 'tsv', 'parquet', 'tiff', 'zarr').

    Returns:
        str: Nested path string, e.g. ``plate1/A1/01/aligned.tiff``.
    """
    dir_parts = [str(v) for v in data_location.values()]
    return str(Path(*dir_parts, f"{info_type}.{file_type}"))


def parse_nested_path(file_path: str, location_keys: list) -> tuple:
    """Parse a nested directory path to extract metadata, info_type, and file_type.

    For a path like ``/output/sbs/images/plate1/A1/01/aligned.tiff``
    with ``location_keys=["plate", "well", "tile"]``, returns::

        ({"plate": "plate1", "well": "A1", "tile": 1}, "aligned", "tiff")

    The last ``len(location_keys)`` directory components above the file are
    mapped to the provided keys, and their values are cast using the data type
    defined in ``FILENAME_METADATA_MAPPING``.

    Args:
        file_path (str): Full or relative file path with nested directory structure.
        location_keys (list of str): Metadata keys corresponding to the directory
            levels directly above the file, from outermost to innermost.
            Must be keys present in ``FILENAME_METADATA_MAPPING``.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Extracted metadata with typed values.
            - info_type (str): The stem of the filename (e.g., 'aligned').
            - file_type (str): The file extension without dot (e.g., 'tiff').
    """
    path = Path(file_path)
    file_type = path.suffix.lstrip(".")
    info_type = path.stem

    dir_parts = list(path.parent.parts)
    n_keys = len(location_keys)

    if len(dir_parts) < n_keys:
        raise ValueError(
            f"Path '{file_path}' has {len(dir_parts)} directory levels but "
            f"{n_keys} location keys were provided: {location_keys}"
        )

    metadata = {}
    for i, key in enumerate(location_keys):
        raw_value = dir_parts[-(n_keys - i)]
        _, data_type = FILENAME_METADATA_MAPPING[key]
        metadata[key] = data_type(raw_value)

    return metadata, info_type, file_type


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


def load_parquet_subset(full_df_fp, n_rows=50000):
    """Load a fixed number of rows from an parquet file without loading entire file into memory.

    Args:
        full_df_fp (str): Path to parquet file.
        n_rows (int): Number of rows to get.

    Returns:
        pd.DataFrame: Subset of the data with combined blocks.
    """
    print(f"Reading first {n_rows:,} rows from {full_df_fp}")

    # read the first n_rows of the file path
    df = ParquetFile(full_df_fp)
    row_subset = next(df.iter_batches(batch_size=n_rows))
    df = pa.Table.from_batches([row_subset]).to_pandas()

    return df


def validate_dtypes(df):
    """Convert DataFrame columns to the most specific data type possible with the following rules.

    - Convert object to bool or string if possible
    - Convert strings to int float if possible
    - Convert floats to int if possible

    Args:
        df : pandas.DataFrame
            The DataFrame to optimize

    Returns:
        pandas.DataFrame
            A new DataFrame with optimized dtypes
    """
    for col in df.columns:
        # Skip columns that are already int
        if pd.api.types.is_integer_dtype(df[col]):
            continue

        # Convert object to bool if possible, else to string
        if pd.api.types.is_object_dtype(df[col]):
            lowered = df[col].dropna().astype(str).str.lower()
            if lowered.isin(["true", "false"]).all():
                df[col] = (
                    df[col].astype(str).str.lower().map({"true": True, "false": False})
                )
            else:
                try:
                    df[col] = df[col].astype("string")
                except ValueError:
                    pass

        # Convert string to float if possible
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

        # Convert float to int if possible
        if pd.api.types.is_float_dtype(df[col]):
            col_nonan = df[col].dropna()
            if len(col_nonan) == 0 or np.allclose(
                col_nonan, col_nonan.round(), rtol=1e-10, atol=1e-10
            ):
                try:
                    df[col] = df[col].astype("Int64")
                except TypeError:
                    pass

    return df


def files_to_tile_mapping(file_paths):
    """Convert list of file paths to tile_id -> file_path mapping.

    Args:
        file_paths (list): List of file paths with tile information in filename

    Returns:
        dict: Mapping from tile_id to file_path
    """
    tile_mapping = {}
    for file_path in file_paths:
        metadata, _, _ = parse_filename(file_path)
        if "tile" in metadata:
            tile_mapping[metadata["tile"]] = str(file_path)
    return tile_mapping


def validate_data_type(data_type):
    """Validate data type parameter.

    Args:
        data_type (str): Data type to validate

    Returns:
        str: Validated data type

    Raises:
        ValueError: If data type is not valid
    """
    valid_types = ["phenotype", "sbs"]
    if data_type not in valid_types:
        raise ValueError(f"data_type must be one of {valid_types}, got '{data_type}'")
    return data_type
