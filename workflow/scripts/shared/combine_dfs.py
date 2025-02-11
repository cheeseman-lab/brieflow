import pandas as pd
from joblib import Parallel, delayed


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Get input, output, and threads from Snakemake
input_files = snakemake.input
output_file = snakemake.output[0]
output_type = getattr(snakemake.params, "output_type", "parquet")
threads = snakemake.threads

# Load, concatenate, and save the data
arr_reads = Parallel(n_jobs=threads)(delayed(get_file)(file) for file in input_files)
df_reads = pd.concat(arr_reads)

# Save the data based on output_type
if output_type == "parquet":
    df_reads.to_parquet(output_file, engine="pyarrow")
elif output_type == "hdf5":
    df_reads.to_hdf(output_file, key="x", mode="w")
elif output_type == "tsv":
    df_reads.to_csv(output_file, sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")
