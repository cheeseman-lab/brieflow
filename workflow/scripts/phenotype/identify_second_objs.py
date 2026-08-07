from tifffile import imread, imwrite
import pandas as pd

from lib.phenotype.segment_secondary_object import segment_second_objs_from_config

# Load input files
data_phenotype = imread(snakemake.input[0])
cells = imread(snakemake.input[1])
cytoplasms = imread(snakemake.input[2])
phenotype_info = pd.read_csv(snakemake.input[3], sep="\t")

# Prepare nuclei centroids from phenotype info (for cell-nucleus distance calculations)
nuclei_centroids_dict = None
if "i" in phenotype_info.columns and "j" in phenotype_info.columns:
    nuclei_centroids_dict = {
        row.get("nuclei_id", idx): (row["i"], row["j"])
        for idx, row in phenotype_info.iterrows()
    }

# Dispatch segmentation (ML or classical) from the config params
second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks = (
    segment_second_objs_from_config(
        image=data_phenotype,
        cell_masks=cells,
        cytoplasm_masks=cytoplasms,
        second_obj_params=snakemake.params.second_obj_params,
        nuclei_centroids=nuclei_centroids_dict,
    )
)

# Save secondary object masks as TIFF
imwrite(snakemake.output[0], second_obj_masks)

# Combine the two tables into one TSV, prefixing columns by table type
cell_summary_df = cell_second_obj_table["cell_summary"]
second_obj_cell_mapping_df = cell_second_obj_table["second_obj_cell_mapping"]
cell_summary_df["table_type"] = "cell_summary"
second_obj_cell_mapping_df["table_type"] = "second_obj_cell_mapping"
cell_summary_df = cell_summary_df.rename(
    columns={
        col: f"cell_summary_{col}"
        for col in cell_summary_df.columns
        if col != "table_type"
    }
)
second_obj_cell_mapping_df = second_obj_cell_mapping_df.rename(
    columns={
        col: f"second_obj_mapping_{col}"
        for col in second_obj_cell_mapping_df.columns
        if col != "table_type"
    }
)
combined_df = pd.concat(
    [cell_summary_df, second_obj_cell_mapping_df], ignore_index=True
)
combined_df.to_csv(snakemake.output[1], sep="\t", index=False)

# Save updated cytoplasm masks as TIFF
imwrite(snakemake.output[2], updated_cytoplasm_masks)
