import pandas as pd

from lib.aggregate.load_format_data import clean_cell_data
from lib.aggregate.feature_processing import feature_transform


# load final merged data
merged_final = pd.read_hdf(snakemake.input[0])

# clean merged data
cleaned_data = clean_cell_data(
    merged_final, snakemake.params["population_feature"], filter_single_gene=False
)
del merged_final

# transform cleaned data
transformations = pd.read_csv(snakemake.params["transformations_fp"], sep="\t")
transformed_data = feature_transform(
    cleaned_data,
    transformations,
    snakemake.params["channels"],
)
del cleaned_data

# save transformed data
transformed_data.to_hdf(snakemake.output[0], key="data", mode="w")
