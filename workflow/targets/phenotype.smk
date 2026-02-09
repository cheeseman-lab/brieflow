from lib.shared.file_utils import get_nested_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


PHENOTYPE_FP = ROOT_FP / "phenotype"

PHENOTYPE_IMG_FMT = IMG_FMT

# determine feature eval outputs based on channel names and segment_cells setting
channel_names = config["phenotype"]["channel_names"]
segment_cells = config["phenotype"].get("segment_cells", True)
prefix = "cell" if segment_cells else "nucleus"
eval_features = [f"{prefix}_{channel}_min" for channel in channel_names]

# --- Conditional path helpers based on image format ---
if PHENOTYPE_IMG_FMT == "zarr":
    from lib.shared.file_utils import get_hcs_nested_path

    _phen_tile_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}", "tile": "{tile}"}
    _phen_well_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}"}
    _phen_plate_loc = {"plate": "{plate}"}

    def _phen_img(info):
        return PHENOTYPE_FP / get_hcs_nested_path(_phen_tile_loc, info)

    def _phen_label(info):
        return PHENOTYPE_FP / get_hcs_nested_path(_phen_tile_loc, info, subdirectory="labels")

    def _phen_tsv(info):
        return PHENOTYPE_FP / "tsvs" / get_nested_path(_phen_tile_loc, info, "tsv")

    def _phen_well_pq(info):
        return PHENOTYPE_FP / "parquets" / get_nested_path(_phen_well_loc, info, "parquet")

    def _phen_plate_eval(subdir, info, ext):
        return PHENOTYPE_FP / "eval" / subdir / get_nested_path(_phen_plate_loc, info, ext)

    # Expansion helpers for rules
    _phen_well_expand = ["row", "col"]
    _phen_tile_expand = ["row", "col", "tile"]
else:
    _phen_tile_loc = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
    _phen_well_loc = {"plate": "{plate}", "well": "{well}"}
    _phen_plate_loc = {"plate": "{plate}"}

    def _phen_img(info):
        return PHENOTYPE_FP / "images" / get_nested_path(_phen_tile_loc, info, PHENOTYPE_IMG_FMT)

    _phen_label = _phen_img  # No labels/ nesting for TIFF

    def _phen_tsv(info):
        return PHENOTYPE_FP / "tsvs" / get_nested_path(_phen_tile_loc, info, "tsv")

    def _phen_well_pq(info):
        return PHENOTYPE_FP / "parquets" / get_nested_path(_phen_well_loc, info, "parquet")

    def _phen_plate_eval(subdir, info, ext):
        return PHENOTYPE_FP / "eval" / subdir / get_nested_path(_phen_plate_loc, info, ext)

    _phen_well_expand = ["well"]
    _phen_tile_expand = ["well", "tile"]


PHENOTYPE_OUTPUTS = {
    "apply_ic_field_phenotype": [_phen_img("illumination_corrected")],
    "align_phenotype": [_phen_img("aligned")],
    "segment_phenotype": [
        _phen_label("nuclei"),
        _phen_label("cells"),
        _phen_tsv("segmentation_stats"),
    ],
    "identify_cytoplasm": [_phen_label("identified_cytoplasms")],
    "extract_phenotype_info": [_phen_tsv("phenotype_info")],
    "combine_phenotype_info": [_phen_well_pq("phenotype_info")],
    "extract_phenotype": [_phen_tsv("phenotype_cp")],
    "merge_phenotype": [
        _phen_well_pq("phenotype_cp"),
        _phen_well_pq("phenotype_cp_min"),
    ],
    "eval_segmentation_phenotype": [
        _phen_plate_eval("segmentation", "segmentation_overview", "tsv"),
        _phen_plate_eval("segmentation", "cell_density_heatmap", "tsv"),
        _phen_plate_eval("segmentation", "cell_density_heatmap", "png"),
    ],
    # create heatmap tsv and png for each evaluated feature
    "eval_features": [
        _phen_plate_eval("features", f"{feature}_heatmap", "tsv")
        for feature in eval_features
    ] + [
        _phen_plate_eval("features", f"{feature}_heatmap", "png")
        for feature in eval_features
    ],
}

# When outputting zarr, image outputs need directory() mapping and should not be temp
# (Snakemake can't reliably temp() a directory output)
# When outputting tiff, intermediate images can be temp() for cleanup
_phenotype_img_temp = directory if PHENOTYPE_IMG_FMT == "zarr" else temp
_phenotype_img_keep = directory if PHENOTYPE_IMG_FMT == "zarr" else None

PHENOTYPE_OUTPUT_MAPPINGS = {
    "apply_ic_field_phenotype": _phenotype_img_temp,
    "align_phenotype": _phenotype_img_keep,
    "segment_phenotype": [_phenotype_img_keep, _phenotype_img_keep, None],
    "identify_cytoplasm": _phenotype_img_temp,
    "extract_phenotype_info": temp,
    "combine_phenotype_info": None,
    "extract_phenotype": temp,
    "merge_phenotype": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

PHENOTYPE_OUTPUTS_MAPPED = map_outputs(PHENOTYPE_OUTPUTS, PHENOTYPE_OUTPUT_MAPPINGS)

PHENOTYPE_TARGETS_ALL = outputs_to_targets(
    PHENOTYPE_OUTPUTS, phenotype_wildcard_combos, PHENOTYPE_OUTPUT_MAPPINGS
)
