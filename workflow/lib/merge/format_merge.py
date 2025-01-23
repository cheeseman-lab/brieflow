import pandas as pd
import numpy as np


def fov_distance(
    df: pd.DataFrame,
    i: str = "i",
    j: str = "j",
    dimensions: tuple = (2960, 2960),
    suffix: str = "",
) -> pd.DataFrame:
    """
    Calculate the distance of each cell from the center of the field of view.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing position coordinates
    i : str
        Column name for x-coordinate, which represents the x-position within that tile
    j : str
        Column name for y-coordinate, which represents the y-position within that tile
    dimensions : tuple
        Tuple of (width, height) for the field of view
    suffix : str
        Suffix to append to the output column name

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'fov_distance{suffix}' column
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
    """
    Determine if a row has a single mapped gene based on gene symbols.

    Parameters
    ----------
    sbs_row : pd.Series
        Single row from an SBS dataframe containing gene symbol columns for genes mapped to each cell
    gene_symbol_0 : str
        Column name of the first gene symbol
    gene_symbol_1 : str
        Column name of the second gene symbol

    Returns
    -------
    bool
        True if only gene_symbol_0 exists or both symbols are identical
    """
    has_single_gene = pd.notnull(sbs_row[gene_symbol_0]) & pd.isnull(
        sbs_row[gene_symbol_1]
    )
    has_matching_genes = sbs_row[gene_symbol_0] == sbs_row[gene_symbol_1]
    return has_single_gene or has_matching_genes


def calculate_channel_mins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimum values across all channel columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing channel columns with '_min' suffix

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'channels_min' column
    """
    min_cols = [col for col in df.columns if "_min" in col]
    df["channels_min"] = df[min_cols].apply(lambda x: x.min(axis=0), axis=1)
    return df
