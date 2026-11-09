import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet

# Load deduplicated merge data (global_i_*/global_j_* already attached in format_merge)
merge_deduplicated = validate_dtypes(read_parquet(snakemake.input[0]))

# Load full feature data
cp_phenotype = validate_dtypes(read_parquet(snakemake.input[1]))

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# Save final merged dataset
write_parquet(merged_final, snakemake.output[0])
