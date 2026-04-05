import numpy as np
import pandas as pd
import anndata as ad

from lib.aggregate.cell_data_utils import load_metadata_cols

# Parameters
metadata_cols_fp = snakemake.params.metadata_cols_fp
use_classifier = snakemake.params.get("use_classifier", False)
control_key = snakemake.params.control_key
perturbation_name_col = snakemake.params.perturbation_name_col
channel_names = snakemake.params.channel_names
channel_combo = snakemake.params.channel_combo

# Load metadata cols from TSV + optional classifier cols
metadata_cols = load_metadata_cols(metadata_cols_fp, use_classifier)

# Load and concatenate all singlecell parquets (one per cell_class)
print("Loading singlecell parquets...")
dfs = []
for path in snakemake.input.singlecell_paths:
    df = pd.read_parquet(path)
    dfs.append(df)
    print(f"  {path}: {df.shape}")

df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {df.shape}")

# Split obs (metadata) and feature columns.
# Any non-numeric column not in metadata_cols is also moved to obs automatically.
non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
obs_cols = list(dict.fromkeys([c for c in metadata_cols if c in df.columns] + non_numeric_cols))
feature_cols = [c for c in df.columns if c not in obs_cols]

print(f"Obs columns: {len(obs_cols)} | Feature columns: {len(feature_cols)}")

# Build obs
obs = df[obs_cols].reset_index(drop=True)

# Add is_control boolean
obs["is_control"] = obs[perturbation_name_col].str.contains(control_key, na=False)

# Build var with structured metadata parsed from feature names
# Feature names follow the pattern: {compartment}_{channel}_{feature_type}
# e.g. nucleus_DAPI_mean, cytoplasm_zernike_9_1, cell_area
channel_names_upper = [c.upper() for c in channel_names]

def parse_feature_name(name):
    parts = name.split("_")
    compartment = parts[0] if parts[0] in ("nucleus", "cytoplasm", "cell") else None
    channel = None
    feature_type = name
    if compartment and len(parts) > 1:
        if parts[1].upper() in channel_names_upper:
            channel = parts[1]
            feature_type = "_".join(parts[2:]) if len(parts) > 2 else ""
        else:
            feature_type = "_".join(parts[1:])
    return compartment, channel, feature_type

parsed = [parse_feature_name(f) for f in feature_cols]
var = pd.DataFrame(
    {
        "compartment": [p[0] for p in parsed],
        "channel": [p[1] for p in parsed],
        "feature_type": [p[2] for p in parsed],
    },
    index=feature_cols,
)

# Build AnnData
X = df[feature_cols].values.astype(np.float32)
adata = ad.AnnData(X=X, obs=obs, var=var)

# Add spatial coordinates from cell centroid columns
if "cell_i" in obs.columns and "cell_j" in obs.columns:
    adata.obsm["spatial"] = obs[["cell_i", "cell_j"]].values.astype(np.float32)
    print("Added obsm['spatial'] from cell_i/cell_j")

# Store pipeline provenance
adata.uns["pipeline"] = {
    "normalization": "zscore",
    "channel_combo": channel_combo,
    "channels": channel_names,
}

print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
