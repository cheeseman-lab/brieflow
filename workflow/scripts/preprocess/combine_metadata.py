"""Script to combine and filter metadata files by well."""

import pandas as pd
from joblib import Parallel, delayed
from lib.shared.file_utils import validate_dtypes, read_tsv_safe

# Load all metadata files
all_dfs = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)

# Combine all dataframes
combined_df = pd.concat([df for df in all_dfs if not df.empty], ignore_index=True)

# Filter by well if well parameter is provided
well_filter = getattr(snakemake.params, "well", None)
if well_filter and "well" in combined_df.columns:
    combined_df = combined_df[combined_df["well"].astype(str) == str(well_filter)]
    print(f"Filtered metadata to {len(combined_df)} rows for well {well_filter}")

# Validate column types
combined_df = validate_dtypes(combined_df)

# Save the data based on file extension
output_path = snakemake.output[0]
if output_path.endswith(".parquet"):
    combined_df.to_parquet(output_path, engine="pyarrow")
elif output_path.endswith(".tsv"):
    combined_df.to_csv(output_path, sep="\t", index=False)
elif output_path.endswith(".csv"):
    combined_df.to_csv(output_path, index=False)
else:
    raise ValueError(f"Unsupported output file extension: {output_path}")
