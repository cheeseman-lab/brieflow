import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet

# Load deduplicated merge data (global_i_*/global_j_* already attached in format_merge)
merge_deduplicated = validate_dtypes(read_parquet(snakemake.input[0]))

approach = snakemake.params.approach

# Load full feature data
cp_phenotype = validate_dtypes(read_parquet(snakemake.input[1]))

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# Rename coordinate columns to global naming convention only for stitch approach
if approach == "stitch":
    merged_final = merged_final.rename(
        columns={
            "i_0": "global_i_0",
            "j_0": "global_j_0",
            "i_1": "global_i_1",
            "j_1": "global_j_1",
        }
    )

# Save final merged dataset
write_parquet(merged_final, snakemake.output[0])
