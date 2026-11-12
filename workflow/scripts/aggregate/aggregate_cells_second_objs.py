from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet
from lib.aggregate.second_obj_utils import (
    aggregate_second_obj_data,
    prefix_second_obj_columns,
)

# Load cell-level data from final_merge
result = validate_dtypes(read_parquet(snakemake.input[0]))

# Get aggregation strategy from config
agg_strategy = snakemake.params.agg_strategy

print(f"Aggregating secondary objects with strategy: {agg_strategy}")
print(f"  Cells: {len(result)} rows")

plate = int(
    snakemake.wildcards.plate
)  # plate is int64 in both merge_final and phenotype parquets
well = str(snakemake.wildcards.well)  # well is always string ("A1", etc.)

# Aggregate each secondary object in turn, one per-object table per object
for name, second_objs_fp in zip(snakemake.params.second_obj_names, snakemake.input[1:]):
    second_objs_df = validate_dtypes(read_parquet(second_objs_fp))
    print(f"  {name}: {len(second_objs_df)} rows")

    # This table is one well's; a well without objects is legal and NaN-filled downstream
    if second_objs_df.empty:
        print(f"  {name}: no objects in this well, skipping")
        continue

    # Filter secondary objects to matching plate/well
    second_objs_filtered = second_objs_df[
        (second_objs_df["plate"] == plate) & (second_objs_df["well"] == well)
    ]

    print(f"  {name} after plate/well filter: {len(second_objs_filtered)} rows")

    # Named objects' tables carry their wildcards; the cell rows already hold row/col
    if name != "second_obj":
        second_objs_filtered = second_objs_filtered.drop(
            columns=["row", "col", "second_obj"], errors="ignore"
        )

    # Named objects prefix their feature columns with the name
    second_objs_filtered = prefix_second_obj_columns(
        second_objs_filtered,
        name,
        ["plate", "well", "tile", "cell_id", "second_obj_id"],
    )

    result = aggregate_second_obj_data(result, second_objs_filtered, agg_strategy)

print(f"  Result: {len(result)} rows, {len(result.columns)} columns")

# Save
write_parquet(result, snakemake.output[0])
