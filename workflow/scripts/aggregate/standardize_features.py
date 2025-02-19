import pandas as pd

from lib.aggregate.feature_processing import grouped_standardization


# load transformed data
transformed_data = pd.read_hdf(snakemake.input[0])

# determine target features
feature_start_idx = transformed_data.columns.get_loc(snakemake.params["feature_start"])
target_features = transformed_data.columns[feature_start_idx:].tolist()

# standardize data
standardized_data = grouped_standardization(
    transformed_data,
    population_feature=snakemake.params.population_feature,
    control_prefix=snakemake.params.control_prefix,
    group_columns=snakemake.params.group_columns,
    index_columns=snakemake.params.index_columns,
    cat_columns=snakemake.params.cat_columns,
    target_features=target_features,
    drop_features=False,
)

# save standardized features
standardized_data.to_hdf(snakemake.output[0], key="x", mode="w")
