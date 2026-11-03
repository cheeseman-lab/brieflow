import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import validate_dtypes, read_tsv_safe
from lib.shared.parquet_io import write_parquet

# Load and concatenate data
all_dfs = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)
valid_dfs = [df for df in all_dfs if not df.empty]
combined_df = (
    pd.concat(valid_dfs).reset_index(drop=True) if valid_dfs else pd.DataFrame()
)

# Validate col types
# Empty dfs can cause issues with dtype
combined_df = validate_dtypes(combined_df)

# Coerce numeric-looking object columns to numeric dtypes
# Prevents schema mismatches (e.g. Int64 vs String) across parquets from different wells
for col in combined_df.select_dtypes(include="object").columns:
    converted = pd.to_numeric(combined_df[col], errors="coerce")
    if converted.notna().sum() >= combined_df[col].notna().sum() * 0.95:
        combined_df[col] = converted

# Save the data based on output_type
output_type = getattr(snakemake.params, "output_type", "parquet")
if output_type == "parquet":
    write_parquet(combined_df, snakemake.output[0])
elif output_type == "tsv":
    combined_df.to_csv(snakemake.output[0], sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")
