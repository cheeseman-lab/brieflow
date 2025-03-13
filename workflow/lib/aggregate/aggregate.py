import numpy as np
import pandas as pd


def aggregate(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    method="mean",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Apply mean or median aggregation to replicate embeddings for each perturbation.
    The function also returns metadata with perturbation labels and cell counts.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aggregated.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str): The column in the metadata containing perturbation information.
        method (str, optional): The aggregation method to use. Must be either "mean" or "median".
            Defaults to "mean".

    Returns:
        tuple:
            - numpy.ndarray: Aggregated embeddings.
            - pandas.DataFrame: Metadata containing perturbation labels and cell counts.
    """
    aggregated_embeddings = []
    aggregated_metadata = []

    metadata = metadata.reset_index(drop=True)
    aggr_func = (
        np.mean if method == "mean" else np.median if method == "median" else None
    )
    if aggr_func is None:
        raise ValueError(f"Invalid aggregation method: {method}")

    grouping = metadata.groupby(pert_col)
    for pert, group in grouping:
        final_emb = aggr_func(embeddings[group.index.values, :], axis=0)
        aggregated_embeddings.append(final_emb)
        aggregated_metadata.append({pert_col: pert, "cell_count": len(group)})

    return np.vstack(aggregated_embeddings), pd.DataFrame(aggregated_metadata)
