import warnings

import pyarrow.dataset as ds
import pandas as pd
import numpy as np

from lib.phenotype.constants import DEFAULT_METADATA_COLS
from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.perturbation_score import get_perturbation_score


warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_selection"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.feature_selection"
)
np.random.seed(0)


cell_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_dataset = cell_dataset.to_table().to_pandas(use_threads=True, memory_pool=None)
original_num_rows = cell_dataset.shape[0]

perturbation_col = cell_dataset[snakemake.params.perturbation_name_col]
perturbed_genes = [
    gene
    for gene in perturbation_col.unique().tolist()
    if not gene.startswith(snakemake.params.control_key)
]

remove_mask = np.zeros(cell_dataset.shape[0], dtype=bool)
nt_idx = perturbation_col.index[
    perturbation_col.str.startswith(snakemake.params.control_key)
].to_numpy()

gene_num = 0
for gene in perturbed_genes:
    print(f"Processing {gene}...")
    gene_idx = perturbation_col.index[perturbation_col == gene].to_numpy()
    nt_keep = np.random.choice(
        nt_idx, size=min(len(gene_idx), len(nt_idx)), replace=False
    )
    keep_idx = np.union1d(gene_idx, nt_keep)
    gene_subset_df = cell_dataset.iloc[keep_idx].copy()
    original_idx = gene_subset_df.index
    gene_subset_df = gene_subset_df.reset_index(drop=True)

    # SCALE PERTURBATION GENE AND CONTROL FEATURES

    metadata_cols = DEFAULT_METADATA_COLS + [
        "class",
        "confidence",
    ]
    feature_cols = gene_subset_df.columns.difference(metadata_cols, sort=False)
    metadata, features = split_cell_data(gene_subset_df, metadata_cols)
    metadata, features = prepare_alignment_data(
        metadata,
        features,
        ["plate", "well"],
        "gene_symbol_0",
        "nontargeting",
        "sgRNA_0",
    )
    features = features.astype(np.float32)
    features = centerscale_on_controls(
        features,
        metadata,
        "gene_symbol_0",
        "nontargeting",
        "batch_values",
    )
    features = pd.DataFrame(features, columns=feature_cols)
    gene_subset_df = pd.concat([metadata, features], axis=1)

    # REMOVE LOW PERTURBATION SCORE CELLS

    perturbation_scores, auc = get_perturbation_score(
        gene_subset_df,
        gene,
        feature_cols,
        perturbation_col="gene_symbol_0",
    )

    # if auc is greater than the cutoff, this perturbation is significant and we want to remove low perturbation cells
    if auc > snakemake.params.auc_cutoff:
        print(f"!! {gene} qualified for filtering with AUC of {auc:.3f}")
        gene_subset_df["perturbation_score"] = perturbation_scores

        gene_subset_df.index = original_idx
        gene_remove_idx = gene_subset_df[
            (gene_subset_df[snakemake.params.perturbation_name_col] == gene)
            & (gene_subset_df["perturbation_score"] <= 0.5)
        ].index
        remove_mask[gene_remove_idx] = True

    gene_num += 1
    if gene_num > 5:
        break

keep_idx_final = np.where(~remove_mask)[0]
num_dropped_rows = len(cell_dataset) - len(keep_idx_final)
print(f"Dropped {num_dropped_rows} low-perturbation cells")

cell_dataset = cell_dataset.iloc[keep_idx_final].reset_index(drop=True)
cell_dataset.to_parquet(snakemake.output[0])
