import pandas as pd

from lib.aggregate.load_format_data import load_hdf_subset
from lib.aggregate.eval_aggregate import plot_feature_distributions, test_missing_values

print("Loading subsets of processed datasets...")
cleaned_data = load_hdf_subset(snakemake.input[0], n_rows=50000)
transformed_data = load_hdf_subset(snakemake.input[1], n_rows=50000)
standardized_data = load_hdf_subset(snakemake.input[2], n_rows=50000)
cell_data = {
    "cleaned": cleaned_data,
    "transformed": transformed_data,
    "standardized": standardized_data,
}

# TODO: once done with Denali testing, use default channels from snakemake, don't force lowercase
channels = [channel.lower() for channel in snakemake.params["channels"]]

# create feature distribution plots for cell and nucleus features
print("Creating feature distribution plots...")

cell_features = ["cell_{}_mean".format(channel) for channel in channels]
cell_features_fig = plot_feature_distributions(
    cell_data, cell_features, remove_clean=True
)
cell_features_fig.savefig(snakemake.output[0])

nucleus_features = ["nucleus_{}_mean".format(channel) for channel in channels]
nucleus_features_fig = plot_feature_distributions(
    cell_data, nucleus_features, remove_clean=True
)
nucleus_features_fig.savefig(snakemake.output[1])


# test for missing values in mitotic, interphase, and all gene data
print("Testing for missing values in final dataframes...")

mitotic_gene_data = pd.read_csv(snakemake.input[3], sep="\t")
mitotic_missing = test_missing_values(mitotic_gene_data, "mitotic")
mitotic_missing.to_csv(snakemake.output[2], sep="\t", index=False)

interphase_gene_data = pd.read_csv(snakemake.input[4], sep="\t")
interphase_missing = test_missing_values(interphase_gene_data, "interphase")
interphase_missing.to_csv(snakemake.output[3], sep="\t", index=False)

all_gene_data = pd.read_csv(snakemake.input[5], sep="\t")
all_missing = test_missing_values(all_gene_data, "all")
all_missing.to_csv(snakemake.output[4], sep="\t", index=False)
