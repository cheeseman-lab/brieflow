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

warnings.filterwarnings("ignore", category=RuntimeWarning)

import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.perturbation_score import (
    get_top_differential_features,
    get_perturbation_scores,
)

np.random.seed(42)

NONTARGETING_CONTROL_MULTIPLIER = 1000
N_FEATURES = 200

pert_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key

filtered_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
perturbation_col = filtered_dataset.to_table(columns=[pert_col]).to_pandas()[pert_col]

filtered_dataset_cols = filtered_dataset.schema.names
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    [
        "class",
        "confidence",
    ],
)
feature_cols = [col for col in filtered_dataset_cols if col not in metadata_cols]

perturbation_scores_col = pd.Series(np.nan, index=perturbation_col.index)
genes = list(perturbation_col.unique())
genes = [gene for gene in genes if not gene.startswith(control_key)]

nontargeting_indices = (
    perturbation_col.str.startswith(control_key).to_numpy().nonzero()[0]
)

for gene in genes:
    gene_indices = (perturbation_col == gene).to_numpy().nonzero()[0]
    if len(gene_indices) < 3:
        print(f"Skipping gene {gene} due to insufficient data")
        continue

    nontargeting_indices = np.random.choice(
        nontargeting_indices,
        size=min(
            len(nontargeting_indices),
            len(gene_indices) * NONTARGETING_CONTROL_MULTIPLIER,
        ),
        replace=False,
    )
    combined_indices = pa.array(np.union1d(gene_indices, nontargeting_indices))

    subset_df = filtered_dataset.scanner(use_threads=True).take(combined_indices)
    subset_df = subset_df.to_pandas(use_threads=True).reset_index(drop=True)

    metadata, features = split_cell_data(subset_df, metadata_cols)
    metadata, features = prepare_alignment_data(
        metadata,
        features,
        snakemake.params.batch_cols,
        pert_col,
        control_key,
        snakemake.params.perturbation_id_col,
    )
    features = features.astype(np.float32)

    features = centerscale_on_controls(
        features,
        metadata,
        pert_col,
        control_key,
        "batch_values",
    )
    features = pd.DataFrame(features, columns=feature_cols)

    subset_df_scaled = pd.concat([metadata, features], axis=1)

    diff_exp_features = get_top_differential_features(
        subset_df_scaled,
        feature_cols,
        gene,
        pert_col,
        control_key,
        n_features=N_FEATURES,
    )

    perturbation_scores = get_perturbation_scores(
        subset_df_scaled, gene, diff_exp_features, pert_col, control_key
    )
    gene_perturbation_scores = perturbation_scores[subset_df_scaled[pert_col] == gene]
    perturbation_scores_col.iloc[gene_indices] = gene_perturbation_scores

    print(f"Average perturbation score for {gene}: {np.mean(gene_perturbation_scores)}")

# set nontargeting perturbation scores to 0
perturbation_scores_col.iloc[nontargeting_indices] = 0

filtered_dataset = filtered_dataset.to_table(use_threads=True).to_pandas()
filtered_dataset.insert(
    len(metadata_cols), "perturbation_score", perturbation_scores_col
)
filtered_dataset.to_parquet(
    snakemake.output[0],
)
