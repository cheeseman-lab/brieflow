import pandas as pd
from joblib import Parallel, delayed

from lib.shared.file_utils import read_tsv_safe

# Load, concatenate, and save the secondary object phenotype data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(read_tsv_safe)(file) for file in snakemake.input
)

# Combine all dataframes, filtering out empty dataframes
valid_dfs = [df for df in arr_reads if not df.empty]
if valid_dfs:
    second_obj_phenotype = pd.concat(valid_dfs)
    print(
        f"Combined {len(valid_dfs)} files with a total of {len(second_obj_phenotype)} secondary object records"
    )
else:
    print("Warning: No valid data files found!")
    second_obj_phenotype = pd.DataFrame()

# Save the combined secondary object phenotype data
second_obj_phenotype.to_parquet(snakemake.output[0])
