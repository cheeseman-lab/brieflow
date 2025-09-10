"""Utilities for aggregating parquet data across multiple dataframes.

This module provides functions to merge cell and vacuole data.
"""

import pandas as pd
import numpy as np
from typing import Literal, List, Dict, Any


def aggregate_vacuole_data(
    cells_df: pd.DataFrame,
    vacuoles_df: pd.DataFrame,
    agg_strategy: Literal["none", "single", "all", "average"],
) -> pd.DataFrame:
    """Aggregate parquet data according to specified strategy.

    This function merges cell-level data with vacuole-level data using different
    strategies to handle the one-to-many relationship between cells and vacuoles.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Pre-loaded cells parquet data. Must contain 'cell_0' column for cell IDs
        and merge keys: 'plate', 'well', 'tile', 'cell_0'
    vacuoles_df : pd.DataFrame
        Pre-loaded vacuoles parquet data. Must contain 'cell_id' column for cell IDs
        and merge keys: 'plate', 'well', 'tile', 'cell_id'
    agg_strategy : {"none", "single", "all", "average"}
        Aggregation strategy:
        - "none": Return cells_df unchanged
        - "single": Add vacuole data only for cells with exactly 1 vacuole
        - "all": Create separate columns for each vacuole (vacuole_area_1, vacuole_area_2, etc.)
        - "average": Average all vacuole measurements per cell

    Returns:
    -------
    pd.DataFrame
        Aggregated dataframe with merged data according to strategy

    Raises:
    ------
    ValueError
        If required merge keys are missing from either dataframe or if strategy is invalid

    """
    # Validate strategy
    valid_strategies = ["none", "single", "all", "average"]
    if agg_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy: {agg_strategy}. Must be one of {valid_strategies}"
        )

    # If strategy is "none", return cells_df as is
    if agg_strategy == "none":
        return cells_df.copy()

    # Define merge keys - handle the cell ID mapping
    cells_merge_keys = ["plate", "well", "tile", "cell_0"]
    vacuoles_merge_keys = ["plate", "well", "tile", "cell_id"]

    # Validate that merge keys exist in dataframes
    _validate_merge_keys(cells_df, cells_merge_keys, "cells")
    _validate_merge_keys(vacuoles_df, vacuoles_merge_keys, "vacuoles")

    # Dispatch to appropriate strategy function
    strategy_functions = {
        "single": _aggregate_single,
        "all": _aggregate_all,
        "average": _aggregate_average,
    }

    return strategy_functions[agg_strategy](
        cells_df.copy(), vacuoles_df, cells_merge_keys, vacuoles_merge_keys
    )


def _validate_merge_keys(
    df: pd.DataFrame, required_keys: List[str], df_name: str
) -> None:
    """Validate that required merge keys exist in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_keys : List[str]
        List of required column names
    df_name : str
        Name of dataframe for error messages

    Raises:
    ------
    ValueError
        If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing merge keys {missing_keys} in {df_name} dataframe")


def _aggregate_single(
    cells_df: pd.DataFrame,
    vacuoles_df: pd.DataFrame,
    cells_merge_keys: List[str],
    vacuoles_merge_keys: List[str],
) -> pd.DataFrame:
    """Single strategy: only add columns for cells with num_vacuoles = 1, NA for others.

    This strategy is useful when you only want to analyze cells that have exactly
    one vacuole, avoiding complications from multiple vacuoles per cell.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell-level data
    vacuoles_df : pd.DataFrame
        Vacuole-level data
    cells_merge_keys : List[str]
        Column names for merging in cells dataframe
    vacuoles_merge_keys : List[str]
        Column names for merging in vacuoles dataframe

    Returns:
    -------
    pd.DataFrame
        Merged data with vacuole columns only populated for single-vacuole cells
    """
    # Filter for cells with exactly 1 vacuole
    single_vacuole_cells = cells_df[cells_df["num_vacuoles"] == 1].copy()

    # Get vacuole columns (excluding merge keys)
    vacuole_cols = [
        col for col in vacuoles_df.columns if col not in vacuoles_merge_keys
    ]

    # Create mapping for merge: rename cell_id to cell_0 in vacuoles for merging
    vacuoles_for_merge = vacuoles_df.copy()
    vacuoles_for_merge = vacuoles_for_merge.rename(columns={"cell_id": "cell_0"})

    # Merge vacuole data for single-vacuole cells
    merged_single = single_vacuole_cells.merge(
        vacuoles_for_merge[cells_merge_keys + vacuole_cols],
        on=cells_merge_keys,
        how="left",
    )

    # For cells with multiple vacuoles or no vacuoles, add NaN columns
    multi_vacuole_cells = cells_df[cells_df["num_vacuoles"] != 1].copy()

    # Create a DataFrame with NaN values for all vacuole columns at once
    if vacuole_cols:
        nan_df = pd.DataFrame(
            np.nan, index=multi_vacuole_cells.index, columns=vacuole_cols
        )
        multi_vacuole_cells = pd.concat([multi_vacuole_cells, nan_df], axis=1)

    # Combine results
    result_df = pd.concat([merged_single, multi_vacuole_cells], ignore_index=True)

    return result_df


def _aggregate_all(
    cells_df: pd.DataFrame,
    vacuoles_df: pd.DataFrame,
    cells_merge_keys: List[str],
    vacuoles_merge_keys: List[str],
) -> pd.DataFrame:
    """All strategy: merge by adding new sets of columns with identifiers.

    This strategy creates separate columns for each vacuole (e.g., vacuole_area_1,
    vacuole_area_2, etc.), allowing analysis of individual vacuoles while
    maintaining the cell-level structure.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell-level data
    vacuoles_df : pd.DataFrame
        Vacuole-level data
    cells_merge_keys : List[str]
        Column names for merging in cells dataframe
    vacuoles_merge_keys : List[str]
        Column names for merging in vacuoles dataframe

    Returns:
    -------
    pd.DataFrame
        Merged data with numbered columns for each vacuole
    """
    # Get vacuole columns (excluding merge keys)
    vacuole_cols = [
        col for col in vacuoles_df.columns if col not in vacuoles_merge_keys
    ]

    # Create mapping for merge: rename cell_id to cell_0 in vacuoles
    vacuoles_numbered = vacuoles_df.copy()
    vacuoles_numbered = vacuoles_numbered.rename(columns={"cell_id": "cell_0"})

    # Group vacuoles by cell and add vacuole number
    vacuoles_numbered["vacuole_num"] = (
        vacuoles_numbered.groupby(cells_merge_keys).cumcount() + 1
    )

    # Get maximum number of vacuoles per cell
    max_vacuoles = vacuoles_numbered.groupby(cells_merge_keys)["vacuole_num"].max()
    max_vacuoles_global = max_vacuoles.max() if len(max_vacuoles) > 0 else 0

    # Start with cells dataframe
    result_df = cells_df.copy()

    # For each vacuole number, create columns and merge
    for vac_num in range(1, max_vacuoles_global + 1):
        # Filter vacuoles for this number
        current_vacuoles = vacuoles_numbered[
            vacuoles_numbered["vacuole_num"] == vac_num
        ].copy()

        # Rename columns to include vacuole number
        col_mapping = {col: f"{col}_{vac_num}" for col in vacuole_cols}
        current_vacuoles = current_vacuoles.rename(columns=col_mapping)

        # Merge with result
        merge_cols = cells_merge_keys + list(col_mapping.values())
        result_df = result_df.merge(
            current_vacuoles[merge_cols], on=cells_merge_keys, how="left"
        )

    return result_df


def _aggregate_average(
    cells_df: pd.DataFrame,
    vacuoles_df: pd.DataFrame,
    cells_merge_keys: List[str],
    vacuoles_merge_keys: List[str],
) -> pd.DataFrame:
    """Average strategy: average all vacuoles corresponding to a cell.

    This strategy computes the mean of all vacuole measurements for each cell,
    providing a single representative value per cell. Numeric columns are averaged,
    while non-numeric columns use the first value.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell-level data
    vacuoles_df : pd.DataFrame
        Vacuole-level data
    cells_merge_keys : List[str]
        Column names for merging in cells dataframe
    vacuoles_merge_keys : List[str]
        Column names for merging in vacuoles dataframe

    Returns:
    -------
    pd.DataFrame
        Merged data with averaged vacuole measurements per cell
    """
    # Get vacuole columns (excluding merge keys)
    vacuole_cols = [
        col for col in vacuoles_df.columns if col not in vacuoles_merge_keys
    ]

    # Create mapping for merge: rename cell_id to cell_0 in vacuoles
    vacuoles_for_merge = vacuoles_df.copy()
    vacuoles_for_merge = vacuoles_for_merge.rename(columns={"cell_id": "cell_0"})

    # Identify numeric columns for averaging
    numeric_cols = (
        vacuoles_for_merge[vacuole_cols]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    non_numeric_cols = [col for col in vacuole_cols if col not in numeric_cols]

    # Group by merge keys and aggregate
    agg_dict = {}

    # Average numeric columns
    for col in numeric_cols:
        agg_dict[col] = "mean"

    # For non-numeric columns, take the first value
    for col in non_numeric_cols:
        agg_dict[col] = "first"

    if agg_dict:  # Only aggregate if there are columns to aggregate
        vacuoles_averaged = (
            vacuoles_for_merge.groupby(cells_merge_keys).agg(agg_dict).reset_index()
        )

        # Merge with cells data
        result_df = cells_df.merge(vacuoles_averaged, on=cells_merge_keys, how="left")
    else:
        # If no vacuole columns to aggregate, just return cells data
        result_df = cells_df.copy()

    return result_df
