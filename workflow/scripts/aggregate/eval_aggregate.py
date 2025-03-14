import pandas as pd

from lib.shared.file_utils import parse_filename, load_parquet_subset
from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

# Load data
class_merge_data = load_parquet_subset(snakemake.input[0], n_rows=20000)
class_gene_data = load_parquet_subset(snakemake.input[1], n_rows=20000)

# Evaluate missing values
nas_df, nas_fig = nas_summary(class_merge_data)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1])

# Evaluate feature distributions
feature_fig = plot_feature_distributions(class_gene_data, "PC_0", num_features=10)
feature_fig.savefig(snakemake.output[2])
