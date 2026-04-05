"""Format cluster results into AnnData h5ad.

Combines perturbation-level aggregated features with PHATE embedding
and cluster assignments into a single AnnData object.
"""

from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd


def parse_feature_metadata(
    feature_cols: List[str],
    channel_names: List[str],
) -> pd.DataFrame:
    """Parse feature column names into structured metadata.

    Infers feature_type, compartment, and channel from the column name pattern:
    - ``nucleus_area`` → shape, nucleus, None
    - ``nucleus_DAPI_mean`` → intensity, nucleus, DAPI
    - ``cell_correlation_CENPA_DAPI`` → correlation, cell, None

    Args:
        feature_cols: List of feature column names.
        channel_names: List of channel names from config.

    Returns:
        DataFrame indexed by feature_id with columns: feature_name,
        feature_type, compartment, channel.
    """
    compartments = {"nucleus", "cell", "cytoplasm"}
    rows = []
    for fid in feature_cols:
        parts = fid.split("_")
        comp = parts[0] if parts[0] in compartments else None
        channel = None

        if comp and len(parts) > 1:
            if parts[1] in channel_names:
                channel = parts[1]

        if comp and channel:
            ftype = "intensity"
        elif comp and len(parts) > 1 and parts[1] == "correlation":
            ftype = "correlation"
        elif comp:
            ftype = "shape"
        else:
            ftype = "other"

        rows.append(
            {
                "feature_name": fid,
                "feature_type": ftype,
                "compartment": comp,
                "channel": channel,
            }
        )

    var = pd.DataFrame(rows, index=feature_cols)
    var.index.name = "feature_id"
    return var


def split_feature_and_metadata_cols(
    df: pd.DataFrame,
    perturbation_col: str,
) -> tuple[List[str], List[str]]:
    """Split DataFrame columns into metadata and feature columns.

    Metadata columns are non-numeric columns plus known metadata keywords.
    Everything else is treated as a feature.

    Args:
        df: Merged DataFrame with all columns.
        perturbation_col: Name of the perturbation identifier column.

    Returns:
        Tuple of (metadata_cols, feature_cols).
    """
    metadata_keywords = {
        perturbation_col,
        "cell_count",
        "perturbation_auc",
        "PHATE_0",
        "PHATE_1",
        "cluster",
        "uniprot_entry",
        "uniprot_function",
        "uniprot_link",
        "mean_potential_to_nontargeting",
        "normalized_potential_to_nontargeting",
        "cell_stage_confidence",
        "col",
        "row",
        "plate",
        "well",
        "tile",
    }
    metadata_cols = [
        c
        for c in df.columns
        if c in metadata_keywords
        or c.endswith("_cluster")
        or not pd.api.types.is_numeric_dtype(df[c])
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    return metadata_cols, feature_cols


def _add_bootstrap_layers(
    adata: ad.AnnData,
    bootstrap_df: pd.DataFrame,
    perturbation_col: str,
    feature_cols: List[str],
) -> None:
    """Add p-value and FDR layers from bootstrap results.

    The bootstrap TSV has columns like ``{feature}_pval``, ``{feature}_log10``,
    ``{feature}_fdr`` for each tested feature. This reshapes them into
    (n_perturbations × n_features) matrices aligned with adata's var index.

    Args:
        adata: AnnData to add layers to (modified in place).
        bootstrap_df: Gene-level bootstrap results with ``gene`` column.
        perturbation_col: Column name used as perturbation identifier.
        feature_cols: Feature column names matching adata.var_names.
    """
    # Bootstrap uses "gene" as the perturbation key
    gene_col = "gene"
    if gene_col not in bootstrap_df.columns:
        return

    # Align bootstrap rows to adata obs order
    bootstrap_df = bootstrap_df.set_index(gene_col)
    common_genes = adata.obs_names.intersection(bootstrap_df.index)
    if len(common_genes) == 0:
        return

    bootstrap_aligned = bootstrap_df.loc[adata.obs_names]

    # Extract p-values and FDR for features that have bootstrap results
    pval_matrix = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
    fdr_matrix = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
    log10_matrix = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)

    for i, feat in enumerate(feature_cols):
        pval_col = f"{feat}_pval"
        fdr_col = f"{feat}_fdr"
        log10_col = f"{feat}_log10"

        if pval_col in bootstrap_aligned.columns:
            pval_matrix[:, i] = bootstrap_aligned[pval_col].values.astype(np.float32)
        if fdr_col in bootstrap_aligned.columns:
            fdr_matrix[:, i] = bootstrap_aligned[fdr_col].values.astype(np.float32)
        if log10_col in bootstrap_aligned.columns:
            log10_matrix[:, i] = bootstrap_aligned[log10_col].values.astype(np.float32)

    # Count how many features have bootstrap data
    n_with_pvals = np.sum(~np.isnan(pval_matrix[0, :]))

    if n_with_pvals > 0:
        adata.layers["p_values"] = pval_matrix
        adata.layers["neg_log10_fdr"] = log10_matrix


def format_cluster_anndata(
    features_genes: pd.DataFrame,
    clustering: pd.DataFrame,
    perturbation_col: str,
    channel_names: List[str],
    cell_class: str,
    channel_combo: str,
    leiden_resolution: str,
    bootstrap_results: Optional[pd.DataFrame] = None,
) -> ad.AnnData:
    """Build AnnData from perturbation-level features and cluster results.

    Args:
        features_genes: Perturbation-level aggregated features (one row per gene).
        clustering: PHATE embedding + cluster assignments.
        perturbation_col: Column name for perturbation identifier.
        channel_names: List of imaging channel names.
        cell_class: Cell class label (e.g. "Interphase").
        channel_combo: Channel combination string.
        leiden_resolution: Leiden clustering resolution.
        bootstrap_results: Optional bootstrap p-values DataFrame.

    Returns:
        AnnData with obs, var, X, obsm, and uns populated.
    """
    # Merge features with clustering
    merged = features_genes.merge(
        clustering,
        on=perturbation_col,
        how="inner",
        suffixes=("", "_cluster"),
    )

    # Split columns
    metadata_cols, feature_cols = split_feature_and_metadata_cols(
        merged, perturbation_col
    )

    # Build obs
    obs = pd.DataFrame(index=merged[perturbation_col].values)
    obs.index.name = "perturbation_id"

    for col in metadata_cols:
        if col != perturbation_col and col in merged.columns:
            obs[col] = merged[col].values

    if "cluster" in obs.columns:
        obs[f"cluster_group_{leiden_resolution}"] = obs.pop("cluster")

    obs["cell_cycle_phase"] = cell_class.lower()

    # Build var
    var = parse_feature_metadata(feature_cols, channel_names)

    # Build X
    X = merged[feature_cols].values.astype(np.float32)

    # Build AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # PHATE embedding
    if "PHATE_0" in merged.columns and "PHATE_1" in merged.columns:
        adata.obsm["X_phate"] = merged[["PHATE_0", "PHATE_1"]].values.astype(np.float32)

    # Percentile rank layer (0–100, per feature across perturbations)
    X_df = pd.DataFrame(X, columns=feature_cols)
    adata.layers["percentile_rank"] = (
        X_df.rank(pct=True).values * 100
    ).astype(np.float32)

    # Bootstrap p-values (optional)
    if bootstrap_results is not None:
        _add_bootstrap_layers(adata, bootstrap_results, perturbation_col, feature_cols)

    # uns metadata
    adata.uns["schema_version"] = "0.1.0"
    adata.uns["default_embedding"] = "X_phate"
    adata.uns["title"] = f"{cell_class} — {channel_combo}"
    adata.uns["channel_combo"] = channel_combo
    adata.uns["cell_class"] = cell_class
    adata.uns["leiden_resolution"] = leiden_resolution
    adata.uns["channels"] = channel_names

    return adata
