"""
## Derive Perturbation Score

We aim to implement a modified version of the perturbation score methodology from [Li et al., 2025](https://www.nature.com/articles/s41556-025-01626-9) on our phenotypic features.
Our implementation is:
1) Identify top n differential features with a two-sample t-test (perturbation vs. nontargeting controls).
2) Compute the perturbation signature β = mean(feature values in gene-perturbed cells) - mean(feature values in controls) for those features.
3) Project and scale:
    - Project every cell onto β to obtain a scalar score.
    - Fit a 1-D linear model to map this projection onto the binary gene label, then rescale the prediction to the [0, 1] interval to yield the final perturbation score.
"""

import warnings

import pyarrow.dataset as ds
import pandas as pd
import numpy as np

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.perturbation_score import (
    get_top_differential_features,
    get_perturbation_scores,
)

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)


N_FEATURES = 200

pert_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key

# Load full filtered dataset
print("Loading dataset perturbations...")
filtered_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
filtered_dataset = filtered_dataset.to_table(use_threads=True).to_pandas()
perturbation_col = filtered_dataset[pert_col]
print(f"Total cells: {len(perturbation_col)}")
print(f"Total unique perturbations: {len(perturbation_col.unique())}")

# determine cols
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    [
        "class",
        "confidence",
    ],
)
feature_cols = [col for col in filtered_dataset.columns.to_list() if col not in metadata_cols]

# centerscale features on controls
metadata, features = split_cell_data(filtered_dataset, metadata_cols)
del filtered_dataset
features = features.astype(np.float32)
metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
)

features = centerscale_on_controls(
    features,
    metadata,
    pert_col,
    control_key,
    "batch_values",
)
scaled_dataset = pd.concat([metadata, pd.DataFrame(features, columns=feature_cols)], axis=1)

perturbation_scores_col = pd.Series(np.nan, index=perturbation_col.index)
genes = sorted([gene for gene in perturbation_col.unique() if not gene.startswith(control_key)])

for gene in genes:
    gene_indices = (perturbation_col == gene).to_numpy().nonzero()[0]
    if len(gene_indices) < 3:
        print(f"Skipping gene {gene} due to insufficient data")
        continue

    nontargeting_indices = (
        perturbation_col.str.startswith(control_key).to_numpy().nonzero()[0]
    )
    nontargeting_indices = np.random.choice(
        nontargeting_indices,
        size=min(
            len(nontargeting_indices),
            len(gene_indices)
        ),
        replace=False,
    )
    
    combined_indices = np.union1d(gene_indices, nontargeting_indices)
    subset_df = scaled_dataset.iloc[combined_indices].reset_index(drop=True)

    diff_exp_features = get_top_differential_features(
        subset_df,
        feature_cols,
        gene,
        pert_col,
        control_key,
        n_features=N_FEATURES,
    )

    perturbation_scores = get_perturbation_scores(
        subset_df, gene, diff_exp_features, pert_col, control_key
    )
    gene_perturbation_scores = perturbation_scores[subset_df[pert_col] == gene]
    perturbation_scores_col.iloc[gene_indices] = gene_perturbation_scores

    print(f"Average perturbation score for {gene}: {np.mean(gene_perturbation_scores)}")

# set nontargeting perturbation scores to 0
perturbation_scores_col.iloc[nontargeting_indices] = 0

# reload filtered dataset
del scaled_dataset
filtered_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
filtered_dataset = filtered_dataset.to_table(use_threads=True).to_pandas()

filtered_dataset.insert(
    len(metadata_cols), "perturbation_score", perturbation_scores_col
)
filtered_dataset.to_parquet(
    snakemake.output[0],
)
