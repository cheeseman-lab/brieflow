import pandas as pd

from lib.merge.eval_merge import plot_channel_histogram

# Load formatted merge data
merge_formatted = pd.read_hdf(snakemake.input[0])

# Cleaned merged dataset
merge_cleaned = merge_formatted.query("channels_min>0")

# If misaligned wells/tiles specified, exclude them
if (
    snakemake.params.misaligned_wells is not None
    and snakemake.params.misaligned_tiles is not None
):
    merge_formatted = merge_formatted.query(
        "well != @misaligned_wells & tile != @misaligned_tiles"
    )

# Generate and save before and after histogram
fig = plot_channel_histogram(merge_formatted, merge_cleaned)
fig.savefig(snakemake.output[0])

# Save cleaned merge data
merge_cleaned.to_hdf(snakemake.output[1], "x", mode="w")
