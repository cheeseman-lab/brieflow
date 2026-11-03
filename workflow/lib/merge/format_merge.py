"""Functions for the format step in the merge process.

This module provides utility functions for formatting and preparing cell data
during the integration of phenotype and SBS datasets. It includes methods to
calculate spatial distances, identify single gene mappings, and compute
channel-specific statistics.

Functions:
    fov_distance: Computes the distance of each cell from the center of the
        field of view to aid in spatial analysis.
    identify_single_gene_mappings: Identifies whether a cell is associated
        with a single gene based on gene symbol columns.
    calculate_channel_mins: Calculates the minimum values across all
        channel columns to assist in downstream analysis.
    attach_global_pixel_coords: Joins per-tile stage metadata onto merged
        cells and computes global pixel coordinates anchored at the
        per-(plate, well) top-left tile corner.

These functions support the formatting of data for subsequent deduplication
and analysis in the merging pipeline.
"""

import pandas as pd
import numpy as np


def fov_distance(
    df: pd.DataFrame,
    i: str = "i",
    j: str = "j",
    dimensions: tuple = (2960, 2960),
    suffix: str = "",
) -> pd.DataFrame:
    """Calculates the distance of each cell from the center of the field of view.

    Args:
        df: DataFrame containing position coordinates.
        i: Column name for the x-coordinate, representing the x-position within the tile. Defaults to "i".
        j: Column name for the y-coordinate, representing the y-position within the tile. Defaults to "j".
        dimensions: Tuple of (width, height) for the field of view. Defaults to (2960, 2960).
        suffix: Suffix to append to the output column name. Defaults to an empty string.

    Returns:
        DataFrame with an additional column 'fov_distance{suffix}' containing the computed distances.
    """
    df[f"fov_distance{suffix}"] = np.sqrt(
        (df[i] - (dimensions[0] / 2)) ** 2 + (df[j] - (dimensions[1] / 2)) ** 2
    )
    return df


def identify_single_gene_mappings(sbs_row: pd.Series) -> bool:
    """Determines if a row has a single mapped gene across all barcode ranks.

    Args:
        sbs_row: Single row containing gene_symbol_{n} columns.

    Returns:
        True if the cell maps to a single unique gene across all ranked barcodes.
    """
    gene_cols = [c for c in sbs_row.index if c.startswith("gene_symbol_")]
    genes = [sbs_row[c] for c in gene_cols if pd.notnull(sbs_row[c])]
    if len(genes) == 0:
        return False
    return len(set(genes)) == 1


def calculate_channel_mins(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates minimum values across all channel columns.

    Args:
        df: DataFrame containing channel columns with '_min' suffix.

    Returns:
        DataFrame with an additional 'channels_min' column.
    """
    min_cols = [col for col in df.columns if "_min" in col]
    df["channels_min"] = df[min_cols].min(axis=1)
    return df


def attach_global_pixel_coords(
    merged: pd.DataFrame,
    metadata: pd.DataFrame,
    tile_dimensions: tuple,
    suffix: str,
) -> pd.DataFrame:
    """Joins per-tile stage metadata onto merged cells and computes global pixel coords.

    Metadata frames carry one row per (plate, well, tile, channel[, cycle]) —
    deduplication on (plate, well, tile) is required to avoid Cartesian
    expansion of the merged frame during the join.

    Args:
        merged: DataFrame with at least (plate, well, tile or site, i_{0/1}, j_{0/1}).
        metadata: DataFrame with x_pos, y_pos, pixel_size_x, pixel_size_y per tile
            (potentially replicated per channel or cycle).
        tile_dimensions: (height, width) of the tile in pixels.
        suffix: "0" for phenotype (joins on tile), "1" for sbs (joins on site).

    Returns:
        DataFrame with global_i_{suffix}, global_j_{suffix} columns added,
        anchored at the per-(plate, well) min tile-corner.
    """
    if metadata is None or len(metadata) == 0:
        return merged

    tile_col_in_merged = "tile" if suffix == "0" else "site"
    meta = (
        metadata[
            ["plate", "well", "tile", "x_pos", "y_pos", "pixel_size_x", "pixel_size_y"]
        ]
        .drop_duplicates(subset=["plate", "well", "tile"])
        .rename(
            columns={
                "tile": tile_col_in_merged,
                "x_pos": f"x_pos_{suffix}",
                "y_pos": f"y_pos_{suffix}",
                "pixel_size_x": f"pixel_size_x_{suffix}",
                "pixel_size_y": f"pixel_size_y_{suffix}",
            }
        )
    )

    merged = merged.merge(meta, how="left", on=["plate", "well", tile_col_in_merged])

    tile_h, tile_w = tile_dimensions
    corner_x_um = (
        merged[f"x_pos_{suffix}"] - tile_w / 2 * merged[f"pixel_size_x_{suffix}"]
    )
    corner_y_um = (
        merged[f"y_pos_{suffix}"] - tile_h / 2 * merged[f"pixel_size_y_{suffix}"]
    )
    cell_x_um = corner_x_um + merged[f"j_{suffix}"] * merged[f"pixel_size_x_{suffix}"]
    cell_y_um = corner_y_um + merged[f"i_{suffix}"] * merged[f"pixel_size_y_{suffix}"]

    well_origin = (
        merged.assign(_corner_x_um=corner_x_um, _corner_y_um=corner_y_um)
        .groupby(["plate", "well"])[["_corner_x_um", "_corner_y_um"]]
        .transform("min")
    )

    merged[f"global_j_{suffix}"] = (
        ((cell_x_um - well_origin["_corner_x_um"]) / merged[f"pixel_size_x_{suffix}"])
        .round()
        .astype("Int32")
    )
    merged[f"global_i_{suffix}"] = (
        ((cell_y_um - well_origin["_corner_y_um"]) / merged[f"pixel_size_y_{suffix}"])
        .round()
        .astype("Int32")
    )

    return merged
