import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import validate_dtypes


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Load and concatenate data
all_dfs = Parallel(n_jobs=snakemake.threads)(
    delayed(get_file)(file) for file in snakemake.input
)
combined_df = pd.concat(all_dfs).reset_index(drop=True)

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
