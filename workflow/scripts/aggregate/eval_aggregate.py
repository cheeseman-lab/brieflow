import random

import pandas as pd
import pyarrow.dataset as ds

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

# Load merge data using PyArrow dataset
merge_data = ds.dataset(snakemake.input.split_datasets_paths, format="parquet")
merge_data = merge_data.to_table(use_threads=True, memory_pool=None).to_pandas()

# Evaluate missing values
nas_df, nas_fig = nas_summary(merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1])

# Load gene data
aligned_data = ds.dataset(snakemake.input[0], format="parquet")
aligned_data = aligned_data.to_table(use_threads=True, memory_pool=None).to_pandas()

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
