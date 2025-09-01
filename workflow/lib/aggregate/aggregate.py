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
    perturbation_score_threshold=0.5,
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
        perturbation_score_threshold (float, optional): Threshold for filtering based on perturbation score.

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

    # filter by perturbation_score threshold; keep NaNs
    mask = metadata["perturbation_score"].isna() | (
        metadata["perturbation_score"] >= perturbation_score_threshold
    )
    metadata = metadata.loc[mask].reset_index(drop=True)
    embeddings = embeddings[mask.to_numpy(), :]

    grouping = metadata.groupby(pert_col)
    for pert, group in grouping:
        final_emb = aggr_func(embeddings[group.index.values, :], axis=0)
        aggregated_embeddings.append(final_emb)

        # aggregate perturbation score with same function
        pert_score = (
            aggr_func(group["perturbation_score"].dropna())
            if not group["perturbation_score"].isna().all()
            else np.nan
        )

        aggregated_metadata.append(
            {
                pert_col: pert,
                "cell_count": len(group),
                "aggregated_perturbation_score": pert_score,
            }
        )

    return np.vstack(aggregated_embeddings), pd.DataFrame(aggregated_metadata)
