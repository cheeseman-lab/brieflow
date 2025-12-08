from tifffile import imread
import pandas as pd

from lib.phenotype.extract_phenotype_second_objs import extract_phenotype_second_objs

# Load inputs
data_phenotype = imread(snakemake.input[0])
second_obj_masks = imread(snakemake.input[1])

# Load only the second_obj_cell_mapping table from the combined dataframe
combined_df = pd.read_csv(snakemake.input[2], sep="\t")
second_obj_cell_mapping_df = combined_df[
    combined_df["table_type"] == "second_obj_cell_mapping"
].copy()

# Create a dictionary to rename columns by removing the 'second_obj_mapping_' prefix
rename_dict = {}
for col in second_obj_cell_mapping_df.columns:
    if col.startswith("second_obj_mapping_"):
        rename_dict[col] = col.replace("second_obj_mapping_", "")

# Rename columns
second_obj_cell_mapping_df = second_obj_cell_mapping_df.rename(columns=rename_dict)

# Get a list of all columns that start with 'cell_summary_'
cell_summary_cols = [
    col for col in second_obj_cell_mapping_df.columns if col.startswith("cell_summary_")
]

# Drop the table_type column and all cell_summary columns
columns_to_drop = ["table_type"] + cell_summary_cols
second_obj_cell_mapping_df = second_obj_cell_mapping_df.drop(columns=columns_to_drop)

# Print the final columns to verify
print("Final columns:", second_obj_cell_mapping_df.columns.tolist())

# Extract secondary object phenotype features
second_obj_phenotype = extract_phenotype_second_objs(
    data_phenotype=data_phenotype,
    second_objs=second_obj_masks,
    wildcards=snakemake.wildcards,
    second_obj_cell_mapping_df=second_obj_cell_mapping_df,
    foci_channel=snakemake.params.foci_channel_index,
    channel_names=snakemake.params.channel_names,
)

# Save results
second_obj_phenotype.to_csv(snakemake.output[0], sep="\t", index=False)
