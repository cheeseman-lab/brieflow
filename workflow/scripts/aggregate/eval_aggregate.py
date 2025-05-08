import random

import numpy as np
import pyarrow.dataset as ds

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

SUBSET_SIZE = 100000

# Get merge dataset
merge_data = ds.dataset(snakemake.input.split_datasets_paths, format="parquet")
# Choose random row indices
total_rows = merge_data.count_rows()
n_sample = min(SUBSET_SIZE, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()
# Load subset
merge_data = merge_data.scanner().take(random_indices)
merge_data = merge_data.to_pandas(use_threads=True, memory_pool=None)

# Evaluate missing values
nas_df, nas_fig = nas_summary(merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1])

# Get aligned dataset
aligned_data = ds.dataset(snakemake.input[0], format="parquet")
# Choose random row indices
total_rows = aligned_data.count_rows()
n_sample = min(SUBSET_SIZE, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()
# Load subset
aligned_data = aligned_data.scanner().take(random_indices)
aligned_data = aligned_data.to_pandas(use_threads=True, memory_pool=None)

# determine original and aligned columns
random.seed(42)
merge_feature_cols = [
    col for col in merge_data.columns if ("cell_" in col and col.endswith("_mean"))
]
pc_cols = [col for col in aligned_data.columns if col.startswith("PC_")]
aligned_feature_cols = random.sample(
    pc_cols, k=min(len(merge_feature_cols), len(pc_cols))
)

# Evaluate feature distributions
feature_distributions_fig = plot_feature_distributions(
    merge_feature_cols,
    merge_data,
    aligned_feature_cols,
    aligned_data,
)
feature_distributions_fig.savefig(snakemake.output[2])
