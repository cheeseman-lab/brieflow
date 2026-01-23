from tifffile import imread, imwrite
import pandas as pd

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

# Get parameters from config
params = snakemake.params.second_obj_params

# Check which segmentation method to use
use_ml = params.get("use_ml_segmentation", False)

# Common parameters shared by both methods
common_params = {
    "image": data_phenotype,
    "second_obj_channel_index": params["second_obj_channel_index"],
    "cell_masks": cells,
    "cytoplasm_masks": cytoplasms,
    "second_obj_min_size": params.get("second_obj_min_size", 10),
    "second_obj_max_size": params.get("second_obj_max_size", 200),
    "size_filter_method": params.get("size_filter_method", "feret"),
    "max_objects_per_cell": params.get("max_objects_per_cell", 120),
    "overlap_threshold": params.get("overlap_threshold", 0.1),
    "nuclei_centroids": nuclei_centroids_dict,
    "max_total_objects": params.get("max_total_objects", 1000),
}

if use_ml:
    from lib.phenotype.segment_secondary_object import segment_second_objs_ml

    # Parameters already handled in common_params (by key name)
    common_param_keys = {
        "second_obj_channel_index",
        "second_obj_min_size",
        "second_obj_max_size",
        "size_filter_method",
        "max_objects_per_cell",
        "overlap_threshold",
        "max_total_objects",
    }
    # Parameters specific to threshold/CV method (should not be passed to ML)
    cv_only_params = {
        "threshold_smoothing_scale",
        "threshold_method",
        "use_morphological_opening",
        "opening_disk_radius",
        "fill_holes",
        "declump_method",
        "declump_mode",
        "suppress_local_maxima",
        "maxima_reduction_factor",
        "use_shape_refinement",
        "proportion_threshold",
    }

    # General config parameters (not segmentation-specific)
    config_level_params = {
        "use_ml_segmentation",
        "second_obj_detection",
        "foci_channel_index",
        "channel_names",
        "dapi_index",
        "cyto_index",
        "align",
        "segmentation_method",
        "reconcile",
        "cp_method",
        "nuclei_diameter",
        "cell_diameter",
        "target",
        "source",
        "riders",
        "remove_channel",
        "upsample_factor",
        "window",
    }
    
    # Collect ML-specific parameters only
    ml_params = {
        k: v
        for k, v in params.items()
        if k not in common_param_keys 
        and k not in cv_only_params 
        and k not in config_level_params
    }

    # Call ML segmentation with common params and ML-specific params
    second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks = (
        segment_second_objs_ml(**common_params, **ml_params)
    )
else:
    from lib.phenotype.segment_secondary_object import segment_second_objs

    # CV-specific parameters with defaults
    cv_params = {
        "threshold_smoothing_scale": params.get("threshold_smoothing_scale", 1.3488),
        "threshold_method": params.get("threshold_method", "otsu_two_peak"),
        "use_morphological_opening": params.get("use_morphological_opening", True),
        "opening_disk_radius": params.get("opening_disk_radius", 1),
        "fill_holes": params.get("fill_holes", "both"),
        "declump_method": params.get("declump_method", "shape"),
        "declump_mode": params.get("declump_mode", "watershed"),
        "suppress_local_maxima": params.get("suppress_local_maxima", 20),
        "maxima_reduction_factor": params.get("maxima_reduction_factor", None),
        "use_shape_refinement": params.get("use_shape_refinement", False),
        "proportion_threshold": params.get("proportion_threshold", 0.4),
    }

    # Call traditional segmentation
    second_obj_masks, cell_second_obj_table, updated_cytoplasm_masks = (
        segment_second_objs(**common_params, **cv_params)
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
