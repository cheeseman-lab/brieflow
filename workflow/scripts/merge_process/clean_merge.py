import pandas as pd

from lib.merge.eval_merge import plot_channel_histogram

# Load formatted merge data
merge_formatted = pd.read_hdf(snakemake.input[0])

# Create and save cleaned merged dataset
merge_cleaned = merge_formatted.query("channels_min>0")
merge_cleaned.to_hdf(snakemake.output[0], "x", mode="w")

# Generate and save before and after histogram
fig = plot_channel_histogram(merge_formatted, merge_cleaned)
fig.savefig(snakemake.output[1])

# TODO: Add all output targets
