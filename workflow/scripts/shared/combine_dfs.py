import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import validate_dtypes, read_tsv_safe

# Load and concatenate data
all_dfs = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)
valid_dfs = [df for df in all_dfs if not df.empty]
combined_df = pd.concat(valid_dfs).reset_index(drop=True) if valid_dfs else pd.DataFrame()

# Validate col types
# Empty dfs can cause issues with dtype
combined_df = validate_dtypes(combined_df)

# Save the data based on output_type
output_type = getattr(snakemake.params, "output_type", "parquet")
if output_type == "parquet":
    combined_df.to_parquet(snakemake.output[0], engine="pyarrow")
elif output_type == "tsv":
    combined_df.to_csv(snakemake.output[0], sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")
