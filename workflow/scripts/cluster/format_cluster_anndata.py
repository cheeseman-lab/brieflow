"""Generate spec-compliant aggregated_data.h5ad for OPS Data Standard.

This is the visualization-level AnnData where:
- obs = perturbations (one row per gene)
- var = standardized feature set
- X = mean aggregated feature values per perturbation
- obsm = PHATE embedding coordinates
- layers = p_values, neg_log10_fdr (optional, from bootstrap)
- uns = schema metadata
"""

import numpy as np
import pandas as pd
import anndata as ad
from itertools import combinations

# --- Load inputs ---

# Perturbation-level aggregated features (mean per gene)
features_genes = pd.read_csv(snakemake.input.features_genes, sep="\t")
print(f"Features genes: {features_genes.shape}")

# Clustering + embedding
clustering = pd.read_csv(snakemake.input.clustering, sep="\t")
print(f"Clustering: {clustering.shape}")

# Parameters
perturbation_col = snakemake.params.perturbation_name_col
channel_names = snakemake.params.channel_names
cell_class = snakemake.wildcards.cell_class
channel_combo = snakemake.wildcards.channel_combo

# --- Define standardized feature set ---
compartments = ["nucleus", "cell"]
shape_measurements = ["area", "eccentricity", "form_factor", "solidity"]
intensity_measurements = [
    "integrated", "mean", "mass_displacement",
    "mean_edge", "std_edge", "mean_frac_0", "mean_frac_3",
]
channel_pairs = list(combinations(sorted(channel_names), 2))

standardized_features = []
for comp in compartments:
    for meas in shape_measurements:
        standardized_features.append(f"{comp}_{meas}")
    for ch in channel_names:
        for meas in intensity_measurements:
            standardized_features.append(f"{comp}_{ch}_{meas}")
    for ch_a, ch_b in channel_pairs:
        standardized_features.append(f"{comp}_correlation_{ch_a}_{ch_b}")

print(f"Standardized features: {len(standardized_features)}")

# --- Match to available features ---
available = set(features_genes.columns)
matched = [f for f in standardized_features if f in available]
missing = [f for f in standardized_features if f not in available]

if missing:
    print(f"WARNING: {len(missing)} standardized features not in data:")
    for m in missing[:10]:
        print(f"  {m}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")

print(f"Matched features: {len(matched)}")

# --- Merge features with clustering ---
merged = features_genes.merge(
    clustering[[perturbation_col, "PHATE_0", "PHATE_1", "cluster", "cell_count"]],
    on=perturbation_col,
    how="inner",
    suffixes=("", "_cluster"),
)
print(f"Merged: {merged.shape}")

# --- Build obs ---
obs = pd.DataFrame(index=merged[perturbation_col].values)
obs.index.name = "perturbation_id"

if "cell_count" in merged.columns:
    obs["cell_count"] = merged["cell_count"].values
if "cluster" in merged.columns:
    obs[f"cluster_group_{snakemake.wildcards.leiden_resolution}"] = (
        merged["cluster"].values
    )
obs["cell_cycle_phase"] = cell_class.lower()

# --- Build var ---
var = pd.DataFrame(index=matched)
var.index.name = "feature_id"

def parse_feature(fid):
    parts = fid.split("_")
    comp = parts[0]
    if len(parts) > 1 and parts[1] == "correlation":
        return "correlation", comp
    elif len(parts) > 1 and parts[1] in channel_names:
        return "intensity", comp
    else:
        return "shape", comp

parsed = [parse_feature(f) for f in matched]
var["feature_name"] = matched
var["feature_type"] = [p[0] for p in parsed]
var["compartment"] = [p[1] for p in parsed]

# --- Build X ---
X = merged[matched].values.astype(np.float32)

# --- Build AnnData ---
adata = ad.AnnData(X=X, obs=obs, var=var)

# Embedding
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

# --- Save ---
print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
