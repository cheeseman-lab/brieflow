"""This module provides functionality for aggregating embeddings based on metadata.

It includes a function to apply mean or median aggregation to replicate embeddings
for each perturbation, along with returning metadata containing perturbation labels
and cell counts.
"""

import numpy as np
import pandas as pd


def aggregate(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    method="mean",
    ps_probability_threshold=None,
    ps_percentile_threshold=None,
    carry_cols: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply mean or median aggregation to replicate embeddings and perturbation scores for each perturbation.

    Rows with perturbation_score below the threshold are dropped (NaNs kept). The function
    returns aggregated embeddings and metadata with perturbation labels, cell counts, and
    aggregated perturbation scores.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aggregated.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str): The column in the metadata containing perturbation information.
        method (str, optional): The aggregation method to use. Must be either "mean" or "median".
            Defaults to "mean".
        ps_probability_threshold (float, optional): Threshold for filtering based on perturbation score.
        ps_percentile_threshold (float, optional): Percentile threshold for filtering based on perturbation score.
        carry_cols (list[str], optional): Metadata columns functionally determined by
            `pert_col` to preserve (one value per group) in the output. Typical use:
            when `pert_col` is a construct ID (e.g. `cell_barcode_0`), carry the
            human-readable gene symbol (`gene_symbol_0`) through so downstream
            clustering / benchmarking / lookups can match by gene name. Raises
            ValueError if a carry_col is missing from metadata or has more than one
            unique value within a group.

    Returns:
        tuple:
            - numpy.ndarray: Aggregated embeddings.
            - pandas.DataFrame: Metadata with perturbation labels, cell counts,
              and aggregated perturbation scores.
    """
    aggregated_embeddings = []
    aggregated_metadata = []

    metadata = metadata.reset_index(drop=True)
    aggr_func = (
        np.mean if method == "mean" else np.median if method == "median" else None
    )
    if aggr_func is None:
        raise ValueError(f"Invalid aggregation method: {method}")

    if carry_cols is None:
        carry_cols = []
    else:
        missing = [c for c in carry_cols if c not in metadata.columns]
        if missing:
            raise ValueError(
                f"carry_cols not found in metadata: {missing}. "
                f"Available columns: {list(metadata.columns)}"
            )
        overlap = [c for c in carry_cols if c == pert_col]
        if overlap:
            raise ValueError(
                f"carry_cols overlap with pert_col: {overlap}. Remove duplicates."
            )

    # filter by ps_probability_threshold; keep NaNs
    if ps_probability_threshold is not None:
        mask = metadata["perturbation_score"].isna() | (
            metadata["perturbation_score"] >= ps_probability_threshold
        )
        metadata = metadata.loc[mask].reset_index(drop=True)
        embeddings = embeddings[mask.to_numpy(), :]

    # filter by ps_percentile_threshold; keep NaNs
    if ps_percentile_threshold is not None:
        threshold_value = np.nanpercentile(
            metadata["perturbation_score"], ps_percentile_threshold * 100
        )
        mask = metadata["perturbation_score"].isna() | (
            metadata["perturbation_score"] >= threshold_value
        )
        metadata = metadata.loc[mask].reset_index(drop=True)
        embeddings = embeddings[mask.to_numpy(), :]

    grouping = metadata.groupby(pert_col)
    for pert, group in grouping:
        final_emb = aggr_func(embeddings[group.index.values, :], axis=0)
        aggregated_embeddings.append(final_emb)

        agg_meta = {
            pert_col: pert,
            "cell_count": len(group),
        }

        # Always include perturbation_auc if present (needed for gene-level filtering in clustering)
        if "perturbation_auc" in metadata.columns:
            agg_meta["perturbation_auc"] = group["perturbation_auc"].iloc[0]

        # Carry through columns that are functionally determined by pert_col.
        for c in carry_cols:
            nuniq = group[c].nunique(dropna=False)
            if nuniq > 1:
                raise ValueError(
                    f"carry_col {c!r} has {nuniq} unique values within group "
                    f"{pert_col}={pert!r}; not functionally determined by {pert_col}."
                )
            agg_meta[c] = group[c].iloc[0]

        if ps_probability_threshold is not None or ps_percentile_threshold is not None:
            # aggregate perturbation score with same function
            pert_score = (
                aggr_func(group["perturbation_score"].dropna())
                if not group["perturbation_score"].isna().all()
                else np.nan
            )

            agg_meta["aggregated_perturbation_score"] = pert_score

        aggregated_metadata.append(agg_meta)

    return np.vstack(aggregated_embeddings), pd.DataFrame(aggregated_metadata)
