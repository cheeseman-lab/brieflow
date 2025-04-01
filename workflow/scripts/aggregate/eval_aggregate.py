import pandas as pd
import pyarrow.dataset as ds

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

# Load gene data
class_aggregated_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load merge data using PyArrow dataset
class_merge_data = ds.dataset(snakemake.input.split_classes_paths, format="parquet")
class_merge_data = class_merge_data.to_table(
    use_threads=True, memory_pool=None
).to_pandas()

# Evaluate missing values
nas_df, nas_fig = nas_summary(class_merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1])

# Evaluate feature distributions
feature_fig = plot_feature_distributions(class_aggregated_data, "PC_0", num_features=10)
feature_fig.savefig(snakemake.output[2])
