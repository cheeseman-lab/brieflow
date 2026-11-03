import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import read_tsv_safe
from lib.shared.parquet_io import write_parquet


# Validate required params
if getattr(snakemake.params, "channel_names", None) is None:
    raise ValueError("Required config parameter 'channel_names' is not set")


# Load, concatenate, and save the phenotype CellProfiler data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)
valid_dfs = [df for df in arr_reads if not df.empty]
phenotype_cp = pd.concat(valid_dfs) if valid_dfs else pd.DataFrame()
write_parquet(phenotype_cp, snakemake.output[0])


# Create subset of features
# Use cell_ prefix if segmenting cells, otherwise nucleus_
segment_cells = snakemake.params.segment_cells
prefix = "cell" if segment_cells else "nucleus"

# Add bounds for each channel
bounds_features = [f"{prefix}_bounds_{i}" for i in range(4)]

# Add minimum intensity feature for each channel
channel_min_features = [
    f"{prefix}_{channel}_min" for channel in snakemake.params.channel_names
]
# Final features
phenotype_cp_min_features = [
    "plate",
    "well",
    "tile",
    "label",
    f"{prefix}_i",
    f"{prefix}_j",
]
phenotype_cp_min_features.extend(bounds_features + channel_min_features)

# Save subset of features
phenotype_cp_min = phenotype_cp[phenotype_cp_min_features]
write_parquet(phenotype_cp_min, snakemake.output[1])
