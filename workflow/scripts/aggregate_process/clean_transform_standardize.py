import pandas as pd

from lib.aggregate.load_format_data import clean_cell_data
from lib.aggregate.feature_processing import feature_transform, grouped_standardization


# load final merged data
merged_final = pd.read_hdf(snakemake.input[0])

# clean merged data
cleaned_data = clean_cell_data(
    merged_final,
    snakemake.params["population_feature"],
    filter_single_gene=snakemake.params["filter_single_gene"],
)
del merged_final
cleaned_data.to_hdf(snakemake.output[0], key="data", mode="w")

# transform cleaned data
transformations = pd.read_csv(snakemake.params["transformations_fp"], sep="\t")
transformed_data = feature_transform(
    cleaned_data,
    transformations,
    snakemake.params["channels"],
)
del cleaned_data
transformed_data.to_hdf(snakemake.output[1], key="data", mode="w")

# determine target features
feature_start_idx = transformed_data.columns.get_loc(snakemake.params["feature_start"])
target_features = transformed_data.columns[feature_start_idx:].tolist()
# standardize data
standardized_data = grouped_standardization(
    transformed_data,
    population_feature=snakemake.params["population_feature"],
    control_prefix=snakemake.params["control_prefix"],
    group_columns=snakemake.params["group_columns"],
    index_columns=snakemake.params["index_columns"],
    cat_columns=snakemake.params["cat_columns"],
    target_features=target_features,
    drop_features=False,
)
del transformed_data

# save standardized features
standardized_data.to_hdf(snakemake.output[2], key="x", mode="w")
