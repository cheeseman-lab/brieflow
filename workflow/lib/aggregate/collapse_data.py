"""This module provides functions for collpasing data during data analysis.

Functions include:
- Collapsing cell-level data to sgRNA-level and gene-level summaries with customizable options.

Functions:
    - collapse_to_sgrna: Aggregate cell-level data to sgRNA-level summaries.
    - collapse_to_gene: Aggregate sgRNA-level data to gene-level summaries.
"""


def collapse_to_sgrna(
    cell_data,
    method="median",
    target_features=None,
    index_features=["gene_symbol_0", "sgRNA_0"],
    control_prefix="sg_nt",
    min_count=None,
):
    """Collapse cell-level data to sgRNA-level summaries.

    Args:
        cell_data (pd.DataFrame): Input dataframe with cell-level data.
        method (str): Method for collapsing ('median' only currently supported).
        target_features (list, optional): Features to collapse. If None, uses all numeric columns.
        index_features (list): Columns that identify sgRNAs.
        control_prefix (str): Prefix identifying control sgRNAs.
        min_count (int, optional): Minimum number of cells required per sgRNA.

    Returns:
        pd.DataFrame: DataFrame with sgRNA-level summaries.
    """
    if target_features is None:
        target_features = [
            col for col in cell_data.columns if col not in index_features
        ]

    if method == "median":
        cell_data_out = (
            cell_data.groupby(index_features)[target_features].median().reset_index()
        )
        cell_data_out["sgrna_count"] = (
            cell_data.groupby(index_features)
            .size()
            .reset_index(name="sgrna_count")["sgrna_count"]
        )

        if min_count is not None:
            cell_data_out = cell_data_out.query("sgrna_count >= @min_count")
    else:
        raise ValueError("Only method='median' is currently supported")

    control_mask = cell_data_out["gene_symbol_0"].str.startswith(control_prefix)
    unique_controls = cell_data_out.loc[control_mask, "gene_symbol_0"].unique()
    if len(unique_controls) == 1:
        print("Multiple control guides not found. Renaming to ensure uniqueness.")
        control_mask = cell_data_out["gene_symbol_0"].str.startswith(control_prefix)
        for idx, row_idx in enumerate(cell_data_out[control_mask].index, 1):
            cell_data_out.loc[row_idx, "gene_symbol_0"] = f"{control_prefix}_{idx}"

    return cell_data_out


def collapse_to_gene(
    sgrna_data, target_features=None, index_features=["gene_symbol_0"], min_count=None
):
    """Collapse sgRNA-level data to gene-level summaries.

    Args:
        sgrna_data (pd.DataFrame): Input dataframe with sgRNA-level data.
        target_features (list, optional): Features to collapse. If None, uses all numeric columns.
        index_features (list): Columns that identify genes.
        min_count (int, optional): Minimum number of sgRNAs required per gene.

    Returns:
        pd.DataFrame: DataFrame with gene-level summaries.
    """
    if target_features is None:
        target_features = [
            col for col in sgrna_data.columns if col not in index_features
        ]

    sgrna_data_out = (
        sgrna_data.groupby(index_features)[target_features].median().reset_index()
    )

    if "sgrna_count" in sgrna_data.columns:
        sgrna_data_out["gene_count"] = (
            sgrna_data.groupby(index_features)["sgrna_count"]
            .sum()
            .reset_index(drop=True)
        )

    if min_count is not None:
        sgrna_data_out = sgrna_data_out.query("gene_count >= @min_count")

    return sgrna_data_out
