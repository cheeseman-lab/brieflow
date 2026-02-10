import pandas as pd
from joblib import Parallel, delayed


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Load, concatenate, and save the phenotype CellProfiler data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(get_file)(file) for file in snakemake.input
)
phenotype_cp = pd.concat(arr_reads)
phenotype_cp.to_parquet(snakemake.output[0])


# Create subset of features
# Use cell_ prefix if segmenting cells, otherwise nucleus_
segment_cells = snakemake.params.get("segment_cells", True)
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
phenotype_cp_min.to_parquet(snakemake.output[1])
