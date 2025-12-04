from tifffile import imread, imwrite
import pandas as pd
import pickle

from lib.phenotype.segment_secondary_object import segment_second_objs

# Load input files
data_phenotype = imread(snakemake.input[0])
cells = imread(snakemake.input[1])
cytoplasms = imread(snakemake.input[2])
phenotype_info = pd.read_csv(snakemake.input[3], sep="\t")

# Prepare nuclei centroids dictionary from phenotype info
nuclei_centroids_dict = None
if "i" in phenotype_info.columns and "j" in phenotype_info.columns:
    nuclei_id_col = (
        "nuclei_id" if "nuclei_id" in phenotype_info.columns else phenotype_info.index
    )
    nuclei_centroids_dict = {
        row.get("nuclei_id", idx): (row["i"], row["j"])
        for idx, row in phenotype_info.iterrows()
    }

# Segment secondary objects
second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks = segment_second_objs(
    image=data_phenotype,
    second_obj_channel_index=snakemake.params.second_obj_channel_index,
    cell_masks=cells,
    cytoplasm_masks=cytoplasms,
    second_obj_min_size=snakemake.params.second_obj_min_size,
    second_obj_max_size=snakemake.params.second_obj_max_size,
    size_filter_method=snakemake.params.size_filter_method,
    threshold_smoothing_scale=snakemake.params.threshold_smoothing_scale,
    threshold_method=snakemake.params.threshold_method,
    use_morphological_opening=snakemake.params.use_morphological_opening,
    opening_disk_radius=snakemake.params.opening_disk_radius,
    fill_holes=snakemake.params.fill_holes,
    declump_method=snakemake.params.declump_method,
    declump_mode=snakemake.params.declump_mode,
    suppress_local_maxima=snakemake.params.suppress_local_maxima,
    maxima_reduction_factor=snakemake.params.maxima_reduction_factor,
    use_shape_refinement=snakemake.params.use_shape_refinement,
    proportion_threshold=snakemake.params.proportion_threshold,
    max_objects_per_cell=snakemake.params.max_objects_per_cell,
    overlap_threshold=snakemake.params.overlap_threshold,
    nuclei_centroids=nuclei_centroids_dict,
    max_total_objects=snakemake.params.max_total_objects,
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
second_obj_cell_mapping_df = second_obj_cell_mapping_df.rename(
    columns=second_obj_mapping_cols
)

# Combine and save
combined_df = pd.concat(
    [cell_summary_df, second_obj_cell_mapping_df], ignore_index=True
)
combined_df.to_csv(snakemake.output[1], sep="\t", index=False)

# Save updated cytoplasm masks as TIFF
imwrite(snakemake.output[2], updated_cytoplasm_masks)
