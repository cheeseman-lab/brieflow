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
    distance = lambda x: np.sqrt(
        (x[i] - (dimensions[0] / 2)) ** 2 + (x[j] - (dimensions[1] / 2)) ** 2
    )
    df[f"fov_distance{suffix}"] = df.apply(distance, axis=1)
    return df


def identify_single_gene_mappings(
    sbs_row: pd.Series,
    gene_symbol_0: str = "gene_symbol_0",
    gene_symbol_1: str = "gene_symbol_1",
) -> bool:
    """Determines if a row has a single mapped gene based on gene symbols.

    Args:
        sbs_row: Single row from an SBS dataframe containing gene symbol columns for genes mapped to each cell.
        gene_symbol_0: Column name of the first gene symbol.
        gene_symbol_1: Column name of the second gene symbol.

    Returns:
        True if only `gene_symbol_0` exists or both symbols are identical.
    """
    has_single_gene = pd.notnull(sbs_row[gene_symbol_0]) & pd.isnull(
        sbs_row[gene_symbol_1]
    )
    has_matching_genes = sbs_row[gene_symbol_0] == sbs_row[gene_symbol_1]
    return has_single_gene or has_matching_genes


def calculate_channel_mins(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates minimum values across all channel columns.

    Args:
        df: DataFrame containing channel columns with '_min' suffix.

    Returns:
        DataFrame with an additional 'channels_min' column.
    """
    min_cols = [col for col in df.columns if "_min" in col]
    df["channels_min"] = df[min_cols].apply(lambda x: x.min(axis=0), axis=1)
    return df
