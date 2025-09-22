import pandas as pd

from lib.shared.file_utils import validate_dtypes

# Load deduplicated merge data
merge_deduplicated = validate_dtypes(pd.read_parquet(snakemake.input[0]))

# Load full feature data
cp_phenotype = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# Rename coordinate columns to global naming convention
coordinate_rename_map = {
    "i_0": "global_i_0",
    "j_0": "global_j_0",
    "i_1": "global_i_1",
    "j_1": "global_j_1",
}

# Only rename columns that exist in the dataframe
existing_columns = merged_final.columns.tolist()
rename_map = {
    old: new for old, new in coordinate_rename_map.items() if old in existing_columns
}

if rename_map:
    merged_final = merged_final.rename(columns=rename_map)

# Save final merged dataset
merged_final.to_parquet(snakemake.output[0])
