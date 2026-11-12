import pandas as pd

from lib.shared.parquet_io import read_parquet, write_parquet
from lib.aggregate.second_obj_utils import prefix_second_obj_columns

# Load the datasets (extract_phenotype_cp writes parquet; identify_second_objs tsv)
phenotype_data = read_parquet(snakemake.input[0])
merged_data = phenotype_data.copy()

# Merge each secondary object's cell summary, one table per object
for name, table_fp in zip(snakemake.params.second_obj_names, snakemake.input[1:]):
    combined_second_obj_df = pd.read_csv(table_fp, sep="\t")
    cell_summary_df = combined_second_obj_df[
        combined_second_obj_df["table_type"] == "cell_summary"
    ].copy()

    if len(phenotype_data) == 0 or len(cell_summary_df) == 0:
        print(f"No {name} cell summary data to merge")
        continue

    # Filter to only keep cell_summary columns (and table_type)
    cell_summary_columns = [
        col
        for col in cell_summary_df.columns
        if col.startswith("cell_summary_") or col == "table_type"
    ]
    cell_summary_df = cell_summary_df[cell_summary_columns]

    # Remove the 'cell_summary_' prefix from column names
    rename_dict = {}
    for col in cell_summary_df.columns:
        if col.startswith("cell_summary_"):
            rename_dict[col] = col.replace("cell_summary_", "")

    cell_summary_df = cell_summary_df.rename(columns=rename_dict)

    # Drop the table_type column
    cell_summary_df = cell_summary_df.drop(columns=["table_type"])

    # Named objects prefix their columns with the name; the flat form keeps today's columns
    cell_summary_df = prefix_second_obj_columns(cell_summary_df, name, ["cell_id"])

    # Merge on cell_id (secondary objects) = label (phenotype)
    merged_data = merged_data.merge(
        cell_summary_df, left_on="label", right_on="cell_id", how="left"
    )

    # Drop the redundant cell_id column
    merged_data = merged_data.drop("cell_id", axis=1)

    print(
        f"Merged {len(phenotype_data)} phenotype records with {len(cell_summary_df)} {name} records"
    )

if len(phenotype_data) == 0:
    # Phenotype data is empty - create an empty DataFrame
    merged_data = pd.DataFrame()
    print("Phenotype dataset is empty - creating empty output")

# Save the merged dataset
write_parquet(merged_data, snakemake.output[0])
print(
    f"Final dataset has {len(merged_data)} rows and {len(merged_data.columns)} columns"
)
