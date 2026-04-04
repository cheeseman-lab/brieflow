"""Format cluster results into AnnData h5ad.

Combines perturbation-level aggregated features with PHATE embedding
and cluster assignments into a single AnnData object:
- obs = perturbation metadata (cell counts, clusters, AUC, uniprot, etc.)
- var = feature columns with parsed metadata (type, compartment, channel)
- X = perturbation-level aggregated feature values
- obsm = PHATE embedding coordinates
- uns = pipeline metadata
"""

import numpy as np
import pandas as pd
import anndata as ad

# --- Load inputs ---

features_genes = pd.read_csv(snakemake.input.features_genes, sep="\t")
print(f"Features genes: {features_genes.shape}")

clustering = pd.read_csv(snakemake.input.clustering, sep="\t")
print(f"Clustering: {clustering.shape}")

# Parameters
perturbation_col = snakemake.params.perturbation_name_col
channel_names = snakemake.params.channel_names
cell_class = snakemake.wildcards.cell_class
channel_combo = snakemake.wildcards.channel_combo
leiden_resolution = snakemake.wildcards.leiden_resolution

# --- Merge on perturbation column ---
merged = features_genes.merge(
    clustering,
    on=perturbation_col,
    how="inner",
    suffixes=("", "_cluster"),
)
print(f"Merged: {merged.shape}")

# --- Identify feature vs metadata columns ---
# Feature columns are numeric and not in the known metadata set
metadata_keywords = [
    perturbation_col, "cell_count", "perturbation_auc",
    "PHATE_0", "PHATE_1", "cluster",
    "uniprot_entry", "uniprot_function", "uniprot_link",
    "mean_potential_to_nontargeting", "normalized_potential_to_nontargeting",
    "cell_stage_confidence", "col", "row", "plate", "well", "tile",
]
# Also exclude any _cluster suffix columns from the merge
metadata_cols = [
    c for c in merged.columns
    if c in metadata_keywords
    or c.endswith("_cluster")
    or not pd.api.types.is_numeric_dtype(merged[c])
]
feature_cols = [c for c in merged.columns if c not in metadata_cols]

print(f"Metadata columns: {len(metadata_cols)}")
print(f"Feature columns: {len(feature_cols)}")

# --- Build obs ---
obs = pd.DataFrame(index=merged[perturbation_col].values)
obs.index.name = "perturbation_id"

for col in metadata_cols:
    if col != perturbation_col and col in merged.columns:
        obs[col] = merged[col].values

# Rename cluster → cluster_group_{resolution}
if "cluster" in obs.columns:
    obs[f"cluster_group_{leiden_resolution}"] = obs.pop("cluster")

# Add cell class
obs["cell_cycle_phase"] = cell_class.lower()

# --- Build var ---
var = pd.DataFrame(index=feature_cols)
var.index.name = "feature_id"

# Parse feature metadata from column names
def parse_feature(fid):
    parts = fid.split("_")
    comp = parts[0] if parts[0] in ("nucleus", "cell", "cytoplasm") else None
    channel = None
    if comp and len(parts) > 1:
        if parts[1] in channel_names:
            channel = parts[1]
    if comp and channel:
        return "intensity", comp, channel
    elif comp and len(parts) > 1 and parts[1] == "correlation":
        return "correlation", comp, None
    elif comp:
        return "shape", comp, None
    else:
        return "other", None, None

parsed = [parse_feature(f) for f in feature_cols]
var["feature_name"] = feature_cols
var["feature_type"] = [p[0] for p in parsed]
var["compartment"] = [p[1] for p in parsed]
var["channel"] = [p[2] for p in parsed]

# --- Build X ---
X = merged[feature_cols].values.astype(np.float32)

# --- Build AnnData ---
adata = ad.AnnData(X=X, obs=obs, var=var)

# PHATE embedding
if "PHATE_0" in merged.columns and "PHATE_1" in merged.columns:
    adata.obsm["X_phate"] = merged[["PHATE_0", "PHATE_1"]].values.astype(np.float32)

# --- Bootstrap p-values (optional) ---
bootstrap_path = snakemake.input.get("bootstrap_results", None)
if bootstrap_path:
    try:
        bootstrap_df = pd.read_csv(bootstrap_path, sep="\t")
        print(f"Bootstrap results loaded: {bootstrap_df.shape}")
        # TODO: reshape into (n_perturbations × n_features) and add as layers
    except Exception as e:
        print(f"Could not load bootstrap results: {e}")

# --- uns metadata ---
adata.uns["schema_version"] = "0.1.0"
adata.uns["default_embedding"] = "X_phate"
adata.uns["title"] = f"{cell_class} — {channel_combo}"
adata.uns["channel_combo"] = channel_combo
adata.uns["cell_class"] = cell_class
adata.uns["leiden_resolution"] = leiden_resolution
adata.uns["channels"] = channel_names

# --- Save ---
print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
