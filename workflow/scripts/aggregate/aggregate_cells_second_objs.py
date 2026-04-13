import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.aggregate.second_obj_utils import aggregate_second_obj_data

# Load cell-level data from final_merge
cells_df = validate_dtypes(pd.read_parquet(snakemake.input[0]))

# Load per-object secondary object data
second_objs_df = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Get aggregation strategy from config
agg_strategy = snakemake.params.agg_strategy

print(f"Aggregating secondary objects with strategy: {agg_strategy}")
print(f"  Cells: {len(cells_df)} rows")
print(f"  Secondary objects: {len(second_objs_df)} rows")

# Filter secondary objects to matching plate/well
plate = str(snakemake.wildcards.plate)
well = str(snakemake.wildcards.well)
second_objs_df["plate"] = second_objs_df["plate"].astype(str)
second_objs_df["well"] = second_objs_df["well"].astype(str)
second_objs_filtered = second_objs_df[
    (second_objs_df["plate"] == plate) & (second_objs_df["well"] == well)
]

print(f"  Secondary objects after plate/well filter: {len(second_objs_filtered)} rows")

# Aggregate
result = aggregate_second_obj_data(cells_df, second_objs_filtered, agg_strategy)

print(f"  Result: {len(result)} rows, {len(result.columns)} columns")

# Save
result.to_parquet(snakemake.output[0])
