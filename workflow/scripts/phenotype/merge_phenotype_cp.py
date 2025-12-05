import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import read_tsv_safe

# Load, concatenate, and save the phenotype CellProfiler data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)
valid_dfs = [df for df in arr_reads if not df.empty]
phenotype_cp = pd.concat(valid_dfs) if valid_dfs else pd.DataFrame()
phenotype_cp.to_parquet(snakemake.output[0])


# Create subset of features
# Add bounds for each channel
bounds_features = [f"cell_bounds_{i}" for i in range(4)]

# Add minimum intensity feature for each channel
channel_min_features = [
    f"cell_{channel}_min" for channel in snakemake.params.channel_names
]
# Final features
phenotype_cp_min_features = [
    "plate",
    "well",
    "tile",
    "label",
    "cell_i",
    "cell_j",
]
phenotype_cp_min_features.extend(bounds_features + channel_min_features)

# Save subset of features
phenotype_cp_min = phenotype_cp[phenotype_cp_min_features]
phenotype_cp_min.to_parquet(snakemake.output[1])
