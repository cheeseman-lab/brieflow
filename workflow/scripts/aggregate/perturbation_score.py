from pathlib import Path

import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.perturbation_score import get_top_differential_features, get_perturbation_score

pert_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key

filtered_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
perturbation_col = filtered_dataset.to_table(columns=["gene_symbol_0"]).to_pandas()["gene_symbol_0"]
genes = list(perturbation_col.unique())

for gene in genes:
    print(f"Processing gene: {gene}")

    gene_indices = perturbation_col.str.contains(gene, na=False).to_numpy().nonzero()[0]
    nontargeting_indices = perturbation_col.str.contains("nontargeting", na=False).to_numpy().nonzero()[0]
    nontargeting_indices = np.random.choice(nontargeting_indices, size=len(gene_indices), replace=False)
    combined_indices = np.union1d(gene_indices, nontargeting_indices)

    subset_df = filtered_dataset.scanner().take(pa.array(combined_indices))
    subset_df = subset_df.to_pandas(use_threads=True, memory_pool=None).reset_index(drop=True)

    metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, True)
    feature_cols = subset_df.columns.difference(metadata_cols, sort=False)

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
        gene, pert_col, control_key,
        n_features=200,
    )

    perturbation_scores = get_perturbation_score(subset_df_scaled, gene, diff_exp_features, pert_col, control_key)
    subset_df_scaled["perturbation_score"] = perturbation_scores
    gene_scored_df = subset_df_scaled[subset_df_scaled["gene_symbol_0"] == gene]

    break

print(gene_scored_df)