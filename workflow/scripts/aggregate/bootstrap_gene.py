"""Bootstrap one gene and each of its constructs, writing only p-values."""

import pandas as pd

from lib.aggregate.bootstrap import (
    bootstrap_seed,
    read_id_keyed_tsv,
    run_gene_bootstrap,
    select_control_pool,
)
from lib.shared.rule_utils import list_bootstrap_constructs

gene_id = snakemake.wildcards.gene
num_sims = snakemake.params.num_sims
perturbation_col = snakemake.params.perturbation_name_col
print(f"Running bootstrap for gene {gene_id} ({num_sims} simulations)")

construct_ids = list_bootstrap_constructs(snakemake.input.bootstrap_data_dir).get(
    gene_id
)
if not construct_ids:
    raise ValueError(f"No construct data found for gene {gene_id}")

construct_features_df = read_id_keyed_tsv(snakemake.input.construct_features_arr)
feature_names = construct_features_df.columns.tolist()
sample_sizes = read_id_keyed_tsv(snakemake.input.sample_sizes)["cell_count"].astype(int)
missing = [c for c in construct_ids if c not in sample_sizes.index]
if missing:
    raise ValueError(f"Constructs not found in sample sizes: {missing}")

controls_df = pd.read_csv(snakemake.input.controls_arr, sep="\t")
construct_controls = [
    select_control_pool(
        controls_df,
        construct_id,
        snakemake.params.get("bootstrap_control_scope", "pooled"),
        reference_group=snakemake.params.get("bootstrap_reference_group", None),
        group_cols=snakemake.params.get("group_cols", None),
        perturbation_id=gene_id,
    ).values
    for construct_id in construct_ids
]

gene_table = pd.read_csv(snakemake.input.gene_table, sep="\t")
gene_row = gene_table[gene_table[perturbation_col].astype(str) == gene_id]
if len(gene_row) == 0:
    raise ValueError(f"Gene {gene_id} not found in gene table")
missing_features = [f for f in feature_names if f not in gene_table.columns]
if missing_features:
    raise ValueError(f"Bootstrap features missing from gene table: {missing_features}")

construct_pvals, gene_pvals = run_gene_bootstrap(
    construct_features_df.loc[construct_ids, feature_names].to_numpy(dtype=float),
    construct_controls,
    sample_sizes.loc[construct_ids].tolist(),
    gene_row[feature_names].to_numpy(dtype=float)[0],
    num_sims,
    bootstrap_seed(gene_id),
)

construct_df = pd.DataFrame(
    {
        "gene": gene_id,
        "construct": construct_ids,
        "sample_size": sample_sizes.loc[construct_ids].to_numpy(),
        "num_sims": num_sims,
    }
)
construct_df[feature_names] = construct_pvals
construct_df.to_csv(snakemake.output.construct_pvals, sep="\t", index=False)

gene_df = pd.DataFrame(
    {
        "gene": [gene_id],
        "num_constructs": [len(construct_ids)],
        "total_cells": [int(gene_row["cell_count"].iloc[0])],
        "num_sims": [num_sims],
    }
)
gene_df[feature_names] = gene_pvals[None, :]
gene_df.to_csv(snakemake.output.gene_pvals, sep="\t", index=False)

print(f"Bootstrap for gene {gene_id} complete: {len(construct_ids)} constructs")
