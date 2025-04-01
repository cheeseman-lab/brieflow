import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.align import (
    prepare_alignment_data,
    embed_by_pca,
    tvn_on_controls,
)

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# Align
features, metadata = prepare_alignment_data(
    cell_data, snakemake.params.batch_cols, snakemake.params.first_feature
)
pca_embeddings = embed_by_pca(
    features.values,
    metadata,
    variance_or_ncomp=snakemake.params.pc_count,
    batch_col="batch_values",
)
tvn_normalized = tvn_on_controls(
    pca_embeddings,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    "batch_values",
)

# Save aligned data
feature_columns = [f"PC_{i}" for i in range(tvn_normalized.shape[1])]
embeddings_df = pd.DataFrame(
    tvn_normalized, index=metadata.index, columns=feature_columns
)
cell_data = pd.concat([metadata, embeddings_df], axis=1)
cell_data.to_parquet(snakemake.output[0], index=False)
