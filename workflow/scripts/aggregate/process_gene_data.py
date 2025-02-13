import pandas as pd

from lib.aggregate.feature_processing import grouped_standardization
from lib.aggregate.collapse_data import collapse_to_sgrna, collapse_to_gene

# load cell data
cell_data = pd.read_hdf(snakemake.input[0])

# determine target features
feature_start_idx = cell_data.columns.get_loc(snakemake.params["feature_start"])
target_features = cell_data.columns[feature_start_idx:].tolist()

# standardize data if indicated
if snakemake.params["standardize_data"]:
    cell_data = grouped_standardization(
        cell_data,
        population_feature=snakemake.params["population_feature"],
        control_prefix=snakemake.params["control_prefix"],
        group_columns=snakemake.params["group_columns"],
        index_columns=snakemake.params["index_columns"],
        cat_columns=snakemake.params["cat_columns"],
        target_features=target_features,
        drop_features=True,
    )

# collapse data to sgrna summaries
sgrna_data = collapse_to_sgrna(
    cell_data,
    method="median",
    target_features=target_features,
    index_features=[snakemake.params["population_feature"], "sgRNA_0"],
    control_prefix=snakemake.params["control_prefix"],
)
del cell_data

# collapse data to gene summaries
gene_data = collapse_to_gene(
    sgrna_data,
    target_features=target_features,
    index_features=[snakemake.params["population_feature"]],
)
del sgrna_data

# save gene summaries
gene_data.to_csv(snakemake.output[0], sep="\t")
