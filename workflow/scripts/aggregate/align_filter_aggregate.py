import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.filter import (
    load_parquet_subset,
    perturbation_filter,
    missing_values_filter,
)
from lib.aggregate.align import (
    prepare_alignment_data,
    embed_by_pca,
    tvn_on_controls,
)
from lib.aggregate.aggregate import aggregate

# Load cell data using PyArrow dataset
cell_data = ds.dataset(snakemake.input[0], format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()

# Filter
perturbation_filtered = perturbation_filter(
    cell_data,
    snakemake.params.perturbation_name_col,
    snakemake.params.perturbation_multi_col,
    snakemake.params.filter_single_pert,
)
missing_values_filtered = missing_values_filter(
    perturbation_filtered,
    snakemake.params.first_feature,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
)

# Align
features, metadata = prepare_alignment_data(
    missing_values_filtered, snakemake.params.batch_cols, snakemake.params.first_feature
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

aggregated_embeddings, aggregated_metadata = aggregate(
    tvn_normalized,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.agg_method,
)

# Aggregate
feature_columns = [f"PC_{i}" for i in range(tvn_normalized.shape[1])]
aggregated_embeddings_df = pd.DataFrame(
    aggregated_embeddings, index=aggregated_metadata.index, columns=feature_columns
)
aggregated_cell_data = (
    pd.concat([aggregated_metadata, aggregated_embeddings_df], axis=1)
    .sort_values("cell_count", ascending=False)
    .reset_index(drop=True)
)

# Save
aggregated_cell_data.to_parquet(snakemake.output[0], index=False)
