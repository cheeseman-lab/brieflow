import pandas as pd

# Load the datasets
phenotype_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load the combined vacuole file and extract cell summary data
combined_vacuole_df = pd.read_csv(snakemake.input[1], sep="\t")
cell_summary_df = combined_vacuole_df[
    combined_vacuole_df["table_type"] == "cell_summary"
].copy()

# Check if we have phenotype data and cell summary data
if len(phenotype_data) > 0 and len(cell_summary_df) > 0:
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

    # Merge on cell_id (vacuoles) = label (phenotype)
    merged_data = phenotype_data.merge(
        cell_summary_df, left_on="label", right_on="cell_id", how="left"
    )

    # Drop the redundant cell_id column
    merged_data = merged_data.drop("cell_id", axis=1)

    print(
        f"Merged {len(phenotype_data)} phenotype records with {len(cell_summary_df)} vacuole records"
    )

elif len(phenotype_data) > 0:
    # No cell summary data available, just use phenotype data
    merged_data = phenotype_data.copy()
    print(
        f"No vacuole data available - using phenotype data only ({len(phenotype_data)} records)"
    )

else:
    # Both datasets are empty - create an empty DataFrame
    merged_data = pd.DataFrame()
    print("Both phenotype and vacuole datasets are empty - creating empty output")

# Save the merged dataset
merged_data.to_csv(snakemake.output[0], sep="\t", index=False)
print(
    f"Final dataset has {len(merged_data)} rows and {len(merged_data.columns)} columns"
)
