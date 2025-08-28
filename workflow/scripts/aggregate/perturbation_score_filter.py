import gc

import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

from lib.phenotype.constants import DEFAULT_METADATA_COLS
from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data

CONTROL_KEY = "nontargeting"

cell_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")

perturbation_col = cell_dataset.to_table(columns=["gene_symbol_0"]).to_pandas()[
    "gene_symbol_0"
]
perturbed_genes = [
    gene
    for gene in perturbation_col.unique().tolist()
    if not gene.startswith(CONTROL_KEY)
]

for gene in perturbed_genes:
    # sample only the gene index and a same number of controls
    gene_idx = perturbation_col.index[
        perturbation_col["gene_symbol_0"] == gene
    ].to_numpy()
    nt_idx = perturbation_col.index[
        perturbation_col["gene_symbol_0"].str.startswith(CONTROL_KEY)
    ].to_numpy()
    nt_keep = np.random.choice(
        nt_idx, size=min(len(gene_idx), len(nt_idx)), replace=False
    )
    keep_idx = np.union1d(gene_idx, nt_keep)
    gene_subset_df = (
        cell_dataset.scanner()
        .take(pa.array(keep_idx))
        .to_pandas(use_threads=True, memory_pool=None)
    )

    print(gene_subset_df.head())

    break
