"""Utilities for aggregating secondary object data into cell-level features.

This module provides functions to merge cell-level data with per-object secondary
object data using configurable aggregation strategies.
"""

import numpy as np
import pandas as pd
from typing import Literal, List


def aggregate_second_obj_data(
    cells_df: pd.DataFrame,
    second_objs_df: pd.DataFrame,
    agg_strategy: Literal["none", "single", "all", "average"],
) -> pd.DataFrame:
    """Aggregate secondary object data according to specified strategy.

    Merges cell-level data with secondary-object-level data using different
    strategies to handle the one-to-many relationship between cells and
    secondary objects.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell-level data from final_merge. Must contain merge keys:
        'plate', 'well', 'tile', 'cell_0'.
    second_objs_df : pd.DataFrame
        Per-object secondary object data from merge_phenotype_second_objs.
        Must contain merge keys: 'plate', 'well', 'tile', 'cell_id'.
    agg_strategy : {"none", "single", "all", "average"}
        Aggregation strategy:
        - "none": Return cells_df unchanged
        - "single": Add features only for cells with exactly 1 secondary object;
          NaN for cells with 0 or 2+ objects (no rows are dropped)
        - "all": Create numbered columns per object (feature_1, feature_2, etc.)
        - "average": Mean of numeric features across all secondary objects per cell

    Returns:
    -------
    pd.DataFrame
        Cell-level data with secondary object features merged according to strategy.
        All cells from cells_df are always preserved.

    Raises:
    ------
    ValueError
        If required merge keys are missing or strategy is invalid.
    """
    valid_strategies = ["none", "single", "all", "average"]
    if agg_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy: {agg_strategy}. Must be one of {valid_strategies}"
        )

    if agg_strategy == "none":
        return cells_df.copy()

    cells_merge_keys = ["plate", "well", "tile", "cell_0"]
    second_objs_merge_keys = ["plate", "well", "tile", "cell_id"]

    _validate_merge_keys(cells_df, cells_merge_keys, "cells")
    _validate_merge_keys(second_objs_df, second_objs_merge_keys, "second_objs")

    strategy_functions = {
        "single": _aggregate_single,
        "all": _aggregate_all,
        "average": _aggregate_average,
    }

    return strategy_functions[agg_strategy](
        cells_df.copy(), second_objs_df, cells_merge_keys, second_objs_merge_keys
    )


def _validate_merge_keys(
    df: pd.DataFrame, required_keys: List[str], df_name: str
) -> None:
    """Validate that required merge keys exist in dataframe."""
    missing_keys = [key for key in required_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing merge keys {missing_keys} in {df_name} dataframe")


def _get_feature_cols(second_objs_df: pd.DataFrame, merge_keys: List[str]) -> List[str]:
    """Get secondary object feature columns (everything except merge keys and second_obj_id)."""
    exclude = set(merge_keys) | {"second_obj_id"}
    return [col for col in second_objs_df.columns if col not in exclude]


def _prepare_second_objs(
    second_objs_df: pd.DataFrame,
    second_objs_merge_keys: List[str],
    cells_merge_keys: List[str],
) -> pd.DataFrame:
    """Rename cell_id to cell_0 for merging with cells dataframe."""
    df = second_objs_df.copy()
    df = df.rename(columns={"cell_id": "cell_0"})
    return df


def _aggregate_single(
    cells_df: pd.DataFrame,
    second_objs_df: pd.DataFrame,
    cells_merge_keys: List[str],
    second_objs_merge_keys: List[str],
) -> pd.DataFrame:
    """Single strategy: populate features only for cells with exactly 1 secondary object.

    All cells are preserved. Cells with 0 or 2+ secondary objects get NaN for
    all secondary object feature columns.
    """
    second_objs = _prepare_second_objs(
        second_objs_df, second_objs_merge_keys, cells_merge_keys
    )
    feature_cols = _get_feature_cols(second_objs_df, second_objs_merge_keys)

    # Count secondary objects per cell
    obj_counts = (
        second_objs.groupby(cells_merge_keys).size().reset_index(name="_obj_count")
    )

    # Tag cells with their object count
    cells_with_count = cells_df.merge(obj_counts, on=cells_merge_keys, how="left")
    cells_with_count["_obj_count"] = cells_with_count["_obj_count"].fillna(0)

    # Split into single-object and other cells
    single_mask = cells_with_count["_obj_count"] == 1
    single_cells = cells_with_count[single_mask].drop(columns=["_obj_count"])
    other_cells = cells_with_count[~single_mask].drop(columns=["_obj_count"])

    # Merge features for single-object cells
    merged_single = single_cells.merge(
        second_objs[cells_merge_keys + feature_cols],
        on=cells_merge_keys,
        how="left",
    )

    # Add NaN columns for other cells
    if feature_cols:
        nan_df = pd.DataFrame(np.nan, index=other_cells.index, columns=feature_cols)
        other_cells = pd.concat([other_cells, nan_df], axis=1)

    return pd.concat([merged_single, other_cells], ignore_index=True)


def _aggregate_all(
    cells_df: pd.DataFrame,
    second_objs_df: pd.DataFrame,
    cells_merge_keys: List[str],
    second_objs_merge_keys: List[str],
) -> pd.DataFrame:
    """All strategy: create numbered columns for each secondary object.

    Creates feature_1, feature_2, etc. columns. Cells with fewer objects than
    the maximum get NaN for the missing object columns.
    """
    second_objs = _prepare_second_objs(
        second_objs_df, second_objs_merge_keys, cells_merge_keys
    )
    feature_cols = _get_feature_cols(second_objs_df, second_objs_merge_keys)

    # Number secondary objects within each cell
    second_objs["_obj_num"] = second_objs.groupby(cells_merge_keys).cumcount() + 1

    max_objs = second_objs["_obj_num"].max() if len(second_objs) > 0 else 0

    result_df = cells_df.copy()

    for obj_num in range(1, max_objs + 1):
        current = second_objs[second_objs["_obj_num"] == obj_num].copy()
        col_mapping = {col: f"{col}_{obj_num}" for col in feature_cols}
        current = current.rename(columns=col_mapping)

        merge_cols = cells_merge_keys + list(col_mapping.values())
        result_df = result_df.merge(
            current[merge_cols], on=cells_merge_keys, how="left"
        )

    return result_df


def _aggregate_average(
    cells_df: pd.DataFrame,
    second_objs_df: pd.DataFrame,
    cells_merge_keys: List[str],
    second_objs_merge_keys: List[str],
) -> pd.DataFrame:
    """Average strategy: mean of numeric features across all secondary objects per cell.

    Non-numeric columns use the first value. Cells with no secondary objects get NaN.
    """
    second_objs = _prepare_second_objs(
        second_objs_df, second_objs_merge_keys, cells_merge_keys
    )
    feature_cols = _get_feature_cols(second_objs_df, second_objs_merge_keys)

    if not feature_cols:
        return cells_df.copy()

    # Identify numeric vs non-numeric feature columns
    numeric_cols = (
        second_objs[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )
    non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = "mean"
    for col in non_numeric_cols:
        agg_dict[col] = "first"

    averaged = second_objs.groupby(cells_merge_keys).agg(agg_dict).reset_index()

    return cells_df.merge(averaged, on=cells_merge_keys, how="left")
