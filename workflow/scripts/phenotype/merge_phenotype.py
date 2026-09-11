from lib.shared.parquet_io import read_parquets, write_parquet


# Validate required params
if getattr(snakemake.params, "channel_names", None) is None:
    raise ValueError("Required config parameter 'channel_names' is not set")


# Load and concatenate the per-tile phenotype CellProfiler parquets.
# read_parquets uses polars scan+concat with a pandas fallback on schema
# mismatch across tiles; it returns an empty DataFrame for an empty input list.
phenotype_cp = read_parquets(list(snakemake.input))
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
