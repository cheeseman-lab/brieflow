import re

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

# Internal pipeline columns dropped from obs. We keep i_0/j_0/i_1/j_1 and
# fov_distance_0/1 (then rename to _phenotype/_sbs below) since they're
# useful for cross-modality QC.
PIPELINE_INTERNAL_COLS = {
    "batch_values",
    "channels_min",
    "site",
}

# _0/_1 in the merge pipeline means dataset_0 = phenotype, dataset_1 = sbs.
# Rename for clarity in the canonical singlecell h5ad. We deliberately do NOT
# rename cell_barcode_*, gene_symbol_*, gene_id_*, Q_min_*, Q_recomb_*,
# no_recomb_*, cell_barcode_peak_*, cell_barcode_count_* — their _0/_1 means
# "top-ranked barcode #0/#1", not phenotype/sbs.
MERGE_DATASET_RENAMES = {
    "cell_0": "cell_phenotype",
    "cell_1": "cell_sbs",
    "i_0": "i_phenotype",
    "j_0": "j_phenotype",
    "i_1": "i_sbs",
    "j_1": "j_sbs",
    "fov_distance_0": "fov_distance_phenotype",
    "fov_distance_1": "fov_distance_sbs",
    "x_pos_0": "x_pos_phenotype",
    "y_pos_0": "y_pos_phenotype",
    "pixel_size_x_0": "pixel_size_x_phenotype",
    "pixel_size_y_0": "pixel_size_y_phenotype",
    "x_pos_1": "x_pos_sbs",
    "y_pos_1": "y_pos_sbs",
    "pixel_size_x_1": "pixel_size_x_sbs",
    "pixel_size_y_1": "pixel_size_y_sbs",
}

# Stage-position columns are intermediate — they were carried through to enable
# global-pixel computation upstream and to expose physical units; keep the
# pixel-size columns in obs (downstream may want to convert pixels↔μm) but drop
# the raw x_pos/y_pos to avoid leaking acquisition-only state.
STAGE_DROP_COLS = {"x_pos_phenotype", "y_pos_phenotype", "x_pos_sbs", "y_pos_sbs"}

# Load metadata cols from TSV + optional classifier cols
metadata_cols = load_metadata_cols(metadata_cols_fp, use_classifier)

# Load and concatenate all filtered parquets (one per cell_class × plate × well).
# These are RAW features (pre-centerscale_on_controls), unlike the previous
# singlecell_paths inputs which were per-batch z-scored.
print("Loading filtered parquets...")
dfs = []
for path in snakemake.input.filtered_paths:
    df = pd.read_parquet(path)
    dfs.append(df)
    print(f"  {path}: {df.shape}")

df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {df.shape}")

# Apply merge-dataset renames before splitting metadata/features. This must
# happen first so all downstream code sees the new names.
present_renames = {k: v for k, v in MERGE_DATASET_RENAMES.items() if k in df.columns}
if present_renames:
    df = df.rename(columns=present_renames)

# Deduplicate to one row per phenotype cell. The merge step can emit several
# rows for the same phenotype mask when multiple SBS detections fall inside
# it; OPS requires globally unique cell_uid, and the canonical "cell" unit
# here is the phenotype mask. Sort by distance ascending so the closest
# SBS match wins, then keep the first per (plate, well, tile, cell_phenotype).
cell_id_col = "cell_phenotype" if "cell_phenotype" in df.columns else "cell_0"
dedup_keys = [c for c in ["plate", "well", "tile", cell_id_col] if c in df.columns]
if "distance" in df.columns:
    df = df.sort_values("distance", kind="stable", na_position="last")
before = len(df)
df = df.drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)
print(f"Dedup on {dedup_keys}: {before} -> {len(df)} cells")

# Reflect the renames in the metadata-cols list so split_cell_data sees them.
metadata_cols = [MERGE_DATASET_RENAMES.get(c, c) for c in metadata_cols]

# Split obs (metadata) and feature columns.
# Any non-numeric column not in metadata_cols is also moved to obs automatically.
non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
obs_cols = list(
    dict.fromkeys([c for c in metadata_cols if c in df.columns] + non_numeric_cols)
)
feature_cols = [c for c in df.columns if c not in obs_cols]

# Build obs, dropping internal pipeline columns
obs = df[obs_cols].reset_index(drop=True)
cols_to_drop = [c for c in PIPELINE_INTERNAL_COLS if c in obs.columns]
if cols_to_drop:
    print(f"Dropping internal pipeline columns from obs: {cols_to_drop}")
    obs = obs.drop(columns=cols_to_drop)

# Add is_control boolean
obs["is_control"] = (
    obs[perturbation_name_col].str.contains(control_key, na=False).astype(bool)
)

# Compose region as plate_well for cross-well grouping.
obs["region"] = obs["plate"].astype(str) + "_" + obs["well"].astype(str)

# Count of ranked barcodes called for each cell. Generalizes over n_barcodes:
# call_cells emits cell_barcode_0, cell_barcode_1, ..., cell_barcode_{n-1}.
bc_cols = sorted(
    (c for c in obs.columns if re.fullmatch(r"cell_barcode_\d+", c)),
    key=lambda c: int(c.rsplit("_", 1)[1]),
)
if bc_cols:
    obs["mapped_n_barcodes"] = sum(obs[c].notna().astype(int) for c in bc_cols)

# Surface global pixel coords under the user-facing global_x / global_y names
# (skimage/brieflow convention: i = row = y, j = col = x). The upstream
# final_merge step computed these in pixels, anchored at per-(plate, well) origin.
if "global_j_0" in obs.columns and "global_i_0" in obs.columns:
    obs["global_x"] = obs["global_j_0"]
    obs["global_y"] = obs["global_i_0"]

# Drop raw stage-center positions but keep pixel sizes for downstream conversion.
stage_drop_present = [c for c in STAGE_DROP_COLS if c in obs.columns]
if stage_drop_present:
    obs = obs.drop(columns=stage_drop_present)

# Add cell_uid as a globally unique cell identifier across experiments.
# Format: {plate}_{well}_{tile}_{cell_phenotype} (was {cell_0} pre-rename).
cell_id_col = "cell_phenotype" if "cell_phenotype" in obs.columns else "cell_0"
obs["cell_uid"] = (
    obs["plate"].astype(str)
    + "_"
    + obs["well"].astype(str)
    + "_"
    + obs["tile"].astype(str)
    + "_"
    + obs[cell_id_col].astype(str)
)
obs = obs.set_index("cell_uid")
print(f"Set obs index to cell_uid (e.g. {obs.index[0]})")

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

# Add spatial coordinates from cell centroid columns (tile-local pixels).
if "cell_i" in obs.columns and "cell_j" in obs.columns:
    adata.obsm["spatial"] = obs[["cell_i", "cell_j"]].values.astype(np.float32)
    print("Added obsm['spatial'] from cell_i/cell_j (tile-local)")

# Add global spatial coordinates (whole-well pixel coords) when available.
if "global_x" in obs.columns and "global_y" in obs.columns:
    adata.obsm["spatial_global"] = (
        obs[["global_x", "global_y"]].fillna(-1).values.astype(np.int32)
    )
    print("Added obsm['spatial_global'] from global_x/global_y")

# Pipeline provenance — features here are raw (no centerscale_on_controls).
adata.uns["pipeline"] = {
    "normalization": "raw",
    "channel_combo": channel_combo,
    "channels": channel_names,
}

for col in adata.obs.columns:
    if adata.obs[col].dtype == object:
        adata.obs[col] = adata.obs[col].fillna("").astype(str)

print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
