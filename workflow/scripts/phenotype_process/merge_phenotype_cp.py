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
phenotype_cp.to_hdf(snakemake.output[0], "x", mode="w")


# Create subset of features
# Add bounds for each channel
bounds_features = [
    f"cell_bounds_{i}" for i in range(len(snakemake.params.channel_names))
]
# Add minimum intensity feature for each channel
channel_min_features = [
    f"cell_{channel}_min" for channel in snakemake.params.channel_names
]
# Final features
phenotype_cp_min_features = [
    "well",
    "tile",
    "label",
    "cell_i",
    "cell_j",
]
phenotype_cp_min_features.extend(bounds_features + channel_min_features)

# Save subset of features
phenotype_cp_min = phenotype_cp[phenotype_cp_min_features]
phenotype_cp_min.to_hdf(snakemake.output[1], "x", mode="w")
