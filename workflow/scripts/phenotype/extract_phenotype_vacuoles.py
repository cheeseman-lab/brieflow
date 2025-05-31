from tifffile import imread
import pandas as pd

from lib.phenotype.extract_phenotype_vacuoles import extract_phenotype_vacuoles

# Load inputs
data_phenotype = imread(snakemake.input[0])
vacuole_masks = imread(snakemake.input[1])

# Load only the vacuole_cell_mapping table from the combined dataframe
combined_df = pd.read_csv(snakemake.input[2], sep="\t")
vacuole_cell_mapping_df = combined_df[
    combined_df["table_type"] == "vacuole_cell_mapping"
].copy()

# Create a dictionary to rename columns by removing the 'vacuole_mapping_' prefix
rename_dict = {}
for col in vacuole_cell_mapping_df.columns:
    if col.startswith("vacuole_mapping_"):
        rename_dict[col] = col.replace("vacuole_mapping_", "")

# Rename columns
vacuole_cell_mapping_df = vacuole_cell_mapping_df.rename(columns=rename_dict)

# Get a list of all columns that start with 'cell_summary_'
cell_summary_cols = [
    col for col in vacuole_cell_mapping_df.columns if col.startswith("cell_summary_")
]

# Drop the table_type column and all cell_summary columns
columns_to_drop = ["table_type"] + cell_summary_cols
vacuole_cell_mapping_df = vacuole_cell_mapping_df.drop(columns=columns_to_drop)

# Print the final columns to verify
print("Final columns:", vacuole_cell_mapping_df.columns.tolist())

# Extract vacuole phenotype features
vacuole_phenotype = extract_phenotype_vacuoles(
    data_phenotype=data_phenotype,
    vacuoles=vacuole_masks,
    vacuole_cell_mapping_df=vacuole_cell_mapping_df,
    wildcards=snakemake.wildcards,
    foci_channel=snakemake.params.foci_channel,
    channel_names=snakemake.params.channel_names,
)

# Save results
vacuole_phenotype.to_csv(snakemake.output[0], sep="\t", index=False)
