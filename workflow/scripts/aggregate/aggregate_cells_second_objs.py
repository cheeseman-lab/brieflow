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

# Screen-level guard: a non-"none" strategy with zero secondary objects screen-wide is a misconfiguration (per-well empties are legal and NaN-filled downstream)
if agg_strategy != "none" and second_objs_df.empty:
    raise ValueError(
        f"aggregate.second_obj_agg_strategy is '{agg_strategy}' but no secondary "
        f"objects were detected anywhere in the screen ({snakemake.input[1]} has 0 "
        f"rows). Set second_obj_agg_strategy: none, or disable "
        f"phenotype.second_obj_detection."
    )

# Filter secondary objects to matching plate/well
plate = int(
    snakemake.wildcards.plate
)  # plate is int64 in both merge_final and phenotype parquets
well = str(snakemake.wildcards.well)  # well is always string ("A1", etc.)
second_objs_filtered = second_objs_df[
    (second_objs_df["plate"] == plate) & (second_objs_df["well"] == well)
]

print(f"  Secondary objects after plate/well filter: {len(second_objs_filtered)} rows")

# Aggregate
result = aggregate_second_obj_data(cells_df, second_objs_filtered, agg_strategy)

print(f"  Result: {len(result)} rows, {len(result.columns)} columns")

# Save
result.to_parquet(snakemake.output[0])
