import pandas as pd
from joblib import Parallel, delayed


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Load, concatenate, and save the vacuole phenotype data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(get_file)(file) for file in snakemake.input
)

# Combine all dataframes, filtering out None values
valid_dfs = [df for df in arr_reads if df is not None]
if valid_dfs:
    vacuole_phenotype = pd.concat(valid_dfs)
    print(
        f"Combined {len(valid_dfs)} files with a total of {len(vacuole_phenotype)} vacuole records"
    )
else:
    print("Warning: No valid data files found!")
    vacuole_phenotype = pd.DataFrame()

# Save the combined vacuole phenotype data
vacuole_phenotype.to_parquet(snakemake.output[0])
