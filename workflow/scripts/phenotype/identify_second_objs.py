from tifffile import imread, imwrite
import pandas as pd
import pickle

from lib.phenotype.segment_secondary_object import segment_second_objs

# Load input files
data_phenotype = imread(snakemake.input[0])
cells = imread(snakemake.input[1])
cytoplasms = imread(snakemake.input[2])
phenotype_info = pd.read_csv(snakemake.input[3], sep="\t")

# Segment secondary objects
second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks = segment_second_objs(
    image=data_phenotype,
    second_obj_channel_index=snakemake.params.second_obj_channel_index,
    cell_masks=cells,
    cytoplasm_masks=cytoplasms,
    second_obj_min_size=snakemake.params.second_obj_min_size,
    second_obj_max_size=snakemake.params.second_obj_max_size,
    nuclei_centroids=phenotype_info,
    suppress_local_maxima=snakemake.params.suppress_local_maxima,
)

# Save outputs
# Save secondary object masks as TIFF
imwrite(snakemake.output[0], second_obj_masks)

# Save cell-secondary object table as TSV
# It has two DataFrames, save both
cell_summary_df = cell_second_obj_table["cell_summary"]
second_obj_cell_mapping_df = cell_second_obj_table["second_obj_cell_mapping"]

# Combine into one DataFrame with a 'table_type' column for filtering
cell_summary_df["table_type"] = "cell_summary"
second_obj_cell_mapping_df["table_type"] = "second_obj_cell_mapping"

# Ensure no column conflicts by prefixing with table type
cell_summary_cols = {
    col: f"cell_summary_{col}" for col in cell_summary_df.columns if col != "table_type"
}
second_obj_mapping_cols = {
    col: f"second_obj_mapping_{col}"
    for col in second_obj_cell_mapping_df.columns
    if col != "table_type"
}

cell_summary_df = cell_summary_df.rename(columns=cell_summary_cols)
second_obj_cell_mapping_df = second_obj_cell_mapping_df.rename(columns=second_obj_mapping_cols)

# Combine and save
combined_df = pd.concat([cell_summary_df, second_obj_cell_mapping_df], ignore_index=True)
combined_df.to_csv(snakemake.output[1], sep="\t", index=False)

# Save updated cytoplasm masks as TIFF
imwrite(snakemake.output[2], updated_cytoplasm_masks)
