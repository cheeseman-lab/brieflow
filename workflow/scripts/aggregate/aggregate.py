import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.aggregate import aggregate

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input[0], format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# Split aligned data into features and metadata
feature_start_idx = cell_data.columns.get_loc("PC_0")
metadata = cell_data.iloc[:, :feature_start_idx].copy()
tvn_normalized = cell_data.iloc[:, feature_start_idx:].to_numpy()

# Aggregate
aggregated_embeddings, aggregated_metadata = aggregate(
    tvn_normalized,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.agg_method,
)

# Save aggregated data
feature_columns = [f"PC_{i}" for i in range(tvn_normalized.shape[1])]
aggregated_embeddings_df = pd.DataFrame(
    aggregated_embeddings, index=aggregated_metadata.index, columns=feature_columns
)
aggregated_cell_data = (
    pd.concat([aggregated_metadata, aggregated_embeddings_df], axis=1)
    .sort_values("cell_count", ascending=False)
    .reset_index(drop=True)
)
aggregated_cell_data.to_csv(snakemake.output[0], sep="\t", index=False)
