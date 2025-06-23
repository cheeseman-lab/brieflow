from tifffile import imread, imwrite
import pandas as pd
import pickle

from lib.phenotype.identify_vacuoles import segment_vacuoles

# Load input files
data_phenotype = imread(snakemake.input[0])
cells = imread(snakemake.input[1])
cytoplasms = imread(snakemake.input[2])
phenotype_info = pd.read_csv(snakemake.input[3], sep="\t")

# Get parameters from snakemake
vacuole_channel_index = snakemake.params.vacuole_channel_index
min_size = snakemake.params.min_size
max_size = snakemake.params.max_size

# Segment vacuoles
vacuole_masks, cell_vacuole_table, updated_cytoplasm_masks = segment_vacuoles(
    image=data_phenotype,
    vacuole_channel_index=vacuole_channel_index,
    nuclei_channel_index=vacuole_channel_index,
    cell_masks=cells,
    cytoplasm_masks=cytoplasms,
    min_diameter=min_diameter,
    max_diameter=max_diameter,
    nuclei_centroids=phenotype_info,
    nuclei_detection=nuclei_detection,
)

# Save outputs
# Save vacuole masks as TIFF
imwrite(snakemake.output[0], vacuole_masks)

# Save cell-vacuole table as TSV
# It has two DataFrames, save both
cell_summary_df = cell_vacuole_table["cell_summary"]
vacuole_cell_mapping_df = cell_vacuole_table["vacuole_cell_mapping"]

# Combine into one DataFrame with a 'table_type' column for filtering
cell_summary_df["table_type"] = "cell_summary"
vacuole_cell_mapping_df["table_type"] = "vacuole_cell_mapping"

# Ensure no column conflicts by prefixing with table type
cell_summary_cols = {
    col: f"cell_summary_{col}" for col in cell_summary_df.columns if col != "table_type"
}
vacuole_mapping_cols = {
    col: f"vacuole_mapping_{col}"
    for col in vacuole_cell_mapping_df.columns
    if col != "table_type"
}

cell_summary_df = cell_summary_df.rename(columns=cell_summary_cols)
vacuole_cell_mapping_df = vacuole_cell_mapping_df.rename(columns=vacuole_mapping_cols)

# Combine and save
combined_df = pd.concat([cell_summary_df, vacuole_cell_mapping_df], ignore_index=True)
combined_df.to_csv(snakemake.output[1], sep="\t", index=False)

# Save updated cytoplasm masks as TIFF
imwrite(snakemake.output[2], updated_cytoplasm_masks)
