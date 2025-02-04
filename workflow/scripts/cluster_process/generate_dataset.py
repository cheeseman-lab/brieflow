import pandas as pd

from lib.cluster.generate_datasets import (
    clean_and_validate,
    split_channels,
    remove_low_number_genes,
    remove_missing_features,
)

# load gene data based on dataset we are creating
if snakemake.params.dataset == "mitotic":
    gene_data = pd.read_csv(snakemake.input[0], sep="\t")
elif snakemake.params.dataset == "interphase":
    gene_data = pd.read_csv(snakemake.input[1], sep="\t")
elif snakemake.params.dataset == "all":
    gene_data = pd.read_csv(snakemake.input[2], sep="\t")
else:
    raise ValueError("Unknown dataset")
print(gene_data.shape)

# clean and validate gene data
validated_data = clean_and_validate(gene_data)

# filter dataset for channels of interest
channel_filtered_data = split_channels(
    validated_data,
    snakemake.params.channel_combo.split("_"),
    snakemake.params.all_channels,
)

# clean low number genes and missing features
cleaned_data = remove_low_number_genes(
    channel_filtered_data, snakemake.params.min_cell_cutoffs[snakemake.params.dataset]
)
cleaned_data = remove_missing_features(cleaned_data)

# save cleaned data
cleaned_data.to_csv(snakemake.output[0], sep="\t", index=False)
