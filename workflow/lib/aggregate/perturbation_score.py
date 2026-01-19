"""This module provides functions for calculating per-cell perturbation scores.

Functions:
- perturbation_score: Process all perturbations and assign scores to cells based on AUC threshold
- calculate_perturbation_scores: Calculate per-cell perturbation scores using logistic regression
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import split_cell_data


def perturbation_score(
    cell_data: pd.DataFrame,
    metadata_cols: list[str],
    perturbation_name_col: str,
    control_key: str,
    minimum_cell_count: int = 100,
    n_jobs: int = -1,
) -> None:
    """Process all perturbations and assign perturbation scores to cells based on AUC threshold.

    This function processes all non-control perturbations in parallel using joblib,
    calculates perturbation scores using logistic regression, and assigns scores to cells.
    The cell_data DataFrame is modified in-place to add 'perturbation_score' and 'perturbation_auc' columns.

    Args:
        cell_data (pd.DataFrame): DataFrame containing cell data that will be modified in-place.
        metadata_cols (list[str]): List of metadata column names that will be updated to include 'perturbation_score'.
        perturbation_name_col (str): Column name containing perturbation identifiers.
        control_key (str): Prefix identifying control perturbations (e.g., 'nontargeting').
        minimum_cell_count (int, optional): Minimum number of cells required to process a perturbation. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs. -1 uses all available CPUs. Defaults to -1.
    """
    perturbation_col = cell_data[perturbation_name_col]
    perturbed_genes = [
        gene
        for gene in perturbation_col.unique().tolist()
        if not gene.startswith(control_key)
    ]
    nt_idx = perturbation_col.index[
        perturbation_col.str.startswith(control_key)
    ].to_numpy()

    print(f"Processing {len(perturbed_genes)} genes with {n_jobs} parallel jobs...")

    # Process genes in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_gene)(
            gene, cell_data, metadata_cols, nt_idx, minimum_cell_count
        )
        for gene in perturbed_genes
    )

    # Collect results and update cell_data
    for result in results:
        if result is not None:
            gene, gene_idx, scores, auc = result
            cell_data.loc[gene_idx, "perturbation_score"] = scores
            cell_data.loc[gene_idx, "perturbation_auc"] = auc


def calculate_perturbation_scores(
    cell_data: pd.DataFrame,
    gene: str,
    feature_cols: list[str],
    perturbation_col: str = "gene_symbol_0",
    n_differential_features: int = 200,
    minimum_cell_count: int = 100,
) -> tuple[pd.Series, float]:
    """Calculate per-cell perturbation scores via 5-fold out-of-fold logistic regression with top-k feature selection.

    AUROC guide:
      - < 0.6  → basically noise; don't filter (return NaN scores and keep all cells)
      - 0.6-0.75 → weak/moderate separation; filter cautiously
      - > 0.75 → decent separation; filtering makes sense
      - > 0.85-0.9 → strong separation; filtering always safe and effective

    Args:
        cell_data (pd.DataFrame): DataFrame containing cell data with features and metadata.
        gene (str): The target gene perturbation to score against.
        feature_cols (list[str]): List of feature column names to use for scoring.
        perturbation_col (str, optional): Column name containing perturbation labels. Defaults to "gene_symbol_0".
        n_differential_features (int, optional): Number of top differential features to select. Defaults to 200.
        minimum_cell_count (int, optional): Minimum number of cells required for scoring. Defaults to 200.

    Returns:
        tuple[pd.Series, float]: A tuple containing the perturbation scores for each cell and the AUC score.
    """
    # if we have too little data, just return NaN scores
    if cell_data.shape[0] < minimum_cell_count:
        return pd.Series(np.nan, index=cell_data.index), np.nan

    y = (cell_data[perturbation_col] == gene).astype(int).to_numpy()
    X_all = cell_data[feature_cols].to_numpy()

    # select top-k differential features (ANOVA F-test)
    k = min(n_differential_features, X_all.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k).fit(X_all, y)
    X = selector.transform(X_all)

    # make number of splits based on cell count
    if cell_data.shape[0] < minimum_cell_count * 2:
        n_splits = 5
    else:
        n_splits = 10

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, scores)

    return pd.Series(scores, index=cell_data.index), auc


def _process_single_gene(
    gene: str,
    cell_data: pd.DataFrame,
    metadata_cols: list[str],
    nt_idx: np.ndarray,
    minimum_cell_count: int,
) -> tuple[str, np.ndarray, pd.Series, float] | None:
    """Process a single gene and return perturbation scores.

    Returns:
        Tuple of (gene, gene_idx, perturbation_scores, auc) or None if skipped.
    """
    perturbation_col = cell_data["gene_symbol_0"]
    gene_idx = perturbation_col.index[perturbation_col == gene].to_numpy()

    # Sample control cells
    rng = np.random.default_rng(hash(gene) % (2**32))
    nt_keep = rng.choice(nt_idx, size=min(len(gene_idx), len(nt_idx)), replace=False)
    keep_idx = np.union1d(gene_idx, nt_keep)
    gene_subset_df = cell_data.iloc[keep_idx].copy()
    original_idx = gene_subset_df.index.copy()
    gene_subset_df = gene_subset_df.reset_index(drop=True)

    if gene_subset_df.shape[0] < minimum_cell_count:
        print(f"!! Skipping {gene} due to low cell count ({gene_subset_df.shape[0]})")
        return None

    # SCALE PERTURBATION GENE AND CONTROL FEATURES
    feature_cols = gene_subset_df.columns.difference(metadata_cols, sort=False)
    metadata, features = split_cell_data(gene_subset_df, metadata_cols)
    metadata, features = prepare_alignment_data(
        metadata,
        features,
        ["plate", "well"],
        "gene_symbol_0",
        "nontargeting",
        "cell_barcode_0",
    )

    features = features.astype(np.float32)
    features = centerscale_on_controls(
        features,
        metadata,
        "gene_symbol_0",
        "nontargeting",
        "batch_values",
    )
    features = pd.DataFrame(features, columns=feature_cols)
    gene_subset_df = pd.concat([metadata, features], axis=1)

    # CALCULATE PERTURBATION SCORES
    perturbation_scores, auc = calculate_perturbation_scores(
        gene_subset_df,
        gene,
        feature_cols,
        perturbation_col="gene_symbol_0",
    )

    print(
        f"!! {gene} perturbation score details| Number of Cells: {gene_subset_df.shape[0] // 2} AUC: {auc:.3f}"
    )

    perturbation_scores.index = original_idx
    return (gene, gene_idx, perturbation_scores[gene_idx], auc)
