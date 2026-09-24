import pandas as pd

from lib.shared.combine_dfs import combine_tile_dfs
from lib.shared.parquet_io import write_parquet

# Read per-tile TSVs, concat, and normalize dtypes (shared helper).
combined_df = combine_tile_dfs(snakemake.input)
# Helper returns None when no tile yielded a frame (all-empty well); snakemake
# still requires output[0] be written, so fall back to an empty frame.
if combined_df is None:
    combined_df = pd.DataFrame()

# Save the data based on output_type
output_type = getattr(snakemake.params, "output_type", "parquet")
if output_type == "parquet":
    write_parquet(combined_df, snakemake.output[0])
elif output_type == "tsv":
    combined_df.to_csv(snakemake.output[0], sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")
