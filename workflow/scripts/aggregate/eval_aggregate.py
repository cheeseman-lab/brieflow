import random

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    }
)

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

SUBSET_SIZE = 100000

# Get merge dataset — filter out empty parquets to avoid schema conflicts
non_empty_split = [
    p for p in snakemake.input.split_datasets_paths if pq.read_metadata(p).num_rows > 0
]

if len(non_empty_split) == 0:
    print("WARNING: No cells in input, writing empty eval outputs")
    import pandas as pd

    pd.DataFrame().to_csv(snakemake.output[0], sep="\t", index=False)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    fig.savefig(snakemake.output[1], dpi=300, bbox_inches="tight")
    fig.savefig(snakemake.output[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    exit(0)

merge_data = ds.dataset(non_empty_split, format="parquet")
total_rows = merge_data.count_rows()
# Choose random row indices
n_sample = min(SUBSET_SIZE, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()
# Load subset
merge_data = merge_data.scanner().take(random_indices)
merge_data = merge_data.to_pandas(use_threads=True, memory_pool=None)

# Evaluate missing values
nas_df, nas_fig = nas_summary(merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1], dpi=300, bbox_inches="tight", transparent=True)

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
# Try compartment prefixes in priority order and use the first that yields
# *_<channel>_mean feature columns. Driven by the wildcard's compartment_combo
# first (whichever compartments this output is actually about), then a fallback
# chain so e.g. second_obj or cytoplasm combos still find intensity means.
compartment_combo = snakemake.wildcards.compartment_combo.split("-")
prefix_priority = compartment_combo + [
    c
    for c in ("cell", "nucleus", "cytoplasm", "second_obj")
    if c not in compartment_combo
]
merge_feature_cols = []
for prefix in prefix_priority:
    merge_feature_cols = [
        col
        for col in merge_data.columns
        if col.startswith(f"{prefix}_") and col.endswith("_mean")
    ]
    if merge_feature_cols:
        print(
            f"[eval] using {len(merge_feature_cols)} '{prefix}_*_mean' feature columns"
        )
        break

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
feature_distributions_fig.savefig(
    snakemake.output[2], dpi=300, bbox_inches="tight", transparent=True
)
