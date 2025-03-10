import pandas as pd

from lib.aggregate.load_format_data import load_parquet_subset
from lib.aggregate.eval_aggregate import (
    plot_feature_distributions,
    test_missing_values,
    calculate_mitotic_percentage,
)

print("Loading subsets of processed datasets...")
cleaned_data = pd.concat(
    [load_parquet_subset(p, n_rows=20000) for p in snakemake.input.cleaned_data_paths],
    ignore_index=True,
)
transformed_data = pd.concat(
    [
        load_parquet_subset(p, n_rows=20000)
        for p in snakemake.input.transformed_data_paths
    ],
    ignore_index=True,
)
standardized_data = pd.concat(
    [
        load_parquet_subset(p, n_rows=20000)
        for p in snakemake.input.standardized_data_paths
    ],
    ignore_index=True,
)
combined_cell_data = {
    "cleaned": cleaned_data,
    "transformed": transformed_data,
    "standardized": standardized_data,
}

# create feature distribution plots for cell and nucleus features
print("Creating feature distribution plots...")

cell_features = [
    "cell_{}_mean".format(channel) for channel in snakemake.params.channels
]
cell_features_fig = plot_feature_distributions(
    combined_cell_data, cell_features, remove_clean=True
)
cell_features_fig.savefig(snakemake.output[0])

nucleus_features = [
    "nucleus_{}_mean".format(channel) for channel in snakemake.params.channels
]
nucleus_features_fig = plot_feature_distributions(
    combined_cell_data, nucleus_features, remove_clean=True
)
nucleus_features_fig.savefig(snakemake.output[1])


# test for missing values in mitotic, interphase, and all gene data
print("Testing for missing values in final dataframes...")

print(snakemake.input.mitotic_gene_data[0])
mitotic_gene_data = pd.read_csv(snakemake.input.mitotic_gene_data[0], sep="\t")
mitotic_missing = test_missing_values(mitotic_gene_data, "mitotic")
mitotic_missing.to_csv(snakemake.output[2], sep="\t", index=False)

interphase_gene_data = pd.read_csv(snakemake.input.interphase_gene_data[0], sep="\t")
interphase_missing = test_missing_values(interphase_gene_data, "interphase")
interphase_missing.to_csv(snakemake.output[3], sep="\t", index=False)

all_gene_data = pd.read_csv(snakemake.input.all_gene_data[0], sep="\t")
all_missing = test_missing_values(all_gene_data, "all")
all_missing.to_csv(snakemake.output[4], sep="\t", index=False)

# calculate mitotic percentages
mitotic_stats = calculate_mitotic_percentage(mitotic_gene_data, interphase_gene_data)
mitotic_stats.to_csv(snakemake.output[5], sep="\t", index=False)
