import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.filter import (
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)
from lib.aggregate.align import (
    prepare_alignment_data,
    embed_by_pca,
    tvn_on_controls,
)
from lib.aggregate.aggregate import aggregate

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input[0], format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# Filter
cell_data = perturbation_filter(
    cell_data,
    snakemake.params.perturbation_name_col,
    snakemake.params.perturbation_multi_col,
    snakemake.params.filter_single_pert,
)
cell_data = missing_values_filter(
    cell_data,
    snakemake.params.first_feature,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
    perturbation_name_col=snakemake.params.perturbation_name_col,
)
cell_data = intensity_filter(
    cell_data,
    snakemake.params.first_feature,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)
# Save filtered data
cell_data.to_parquet(snakemake.output[0], index=False)


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
cell_data.to_parquet(snakemake.output[1], index=False)

# Aggregate
aggregated_embeddings, aggregated_metadata = aggregate(
    tvn_normalized,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.agg_method,
)

# Save aggregated data
aggregated_embeddings_df = pd.DataFrame(
    aggregated_embeddings, index=aggregated_metadata.index, columns=feature_columns
)
aggregated_cell_data = (
    pd.concat([aggregated_metadata, aggregated_embeddings_df], axis=1)
    .sort_values("cell_count", ascending=False)
    .reset_index(drop=True)
)
aggregated_cell_data.to_csv(snakemake.output[2], sep="\t", index=False)
