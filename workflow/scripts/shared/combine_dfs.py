import pandas as pd

from lib.shared.combine_dfs import combine_tile_dfs
from lib.shared.parquet_io import write_parquet

# Concatenate the per-tile tables; None means no tile had rows, but the output must exist
combined_df = combine_tile_dfs(snakemake.input)
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
