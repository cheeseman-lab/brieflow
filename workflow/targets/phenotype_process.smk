from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


PHENOTYPE_PROCESS_FP = ROOT_FP / "phenotype_process"

# determine feature eval outputs based on channel names
channel_names = config["phenotype_process"]["channel_names"]
eval_features = [f"cell_{channel}_min" for channel in channel_names]

PHENOTYPE_PROCESS_OUTPUTS = {
    "apply_ic_field_phenotype": [
        PHENOTYPE_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    ],
    "segment_phenotype": [
        PHENOTYPE_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"),
        PHENOTYPE_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tiff"),
        PHENOTYPE_PROCESS_FP
        / "tsvs"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "segmentation_stats", "tsv"
        ),
    ],
    "identify_cytoplasm": [
        PHENOTYPE_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "identified_cytoplasms", "tiff"
        ),
    ],
    "extract_phenotype_info": [
        PHENOTYPE_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "phenotype_info", "tsv"),
    ],
    "merge_phenotype_info": [
        PHENOTYPE_PROCESS_FP / "hdfs" / get_filename({}, "phenotype_info", "hdf5"),
    ],
    "extract_phenotype_cp": [
        PHENOTYPE_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "phenotype_cp", "tsv"),
    ],
    "merge_phenotype_cp": [
        PHENOTYPE_PROCESS_FP / "hdfs" / get_filename({}, "phenotype_cp", "hdf5"),
        PHENOTYPE_PROCESS_FP / "hdfs" / get_filename({}, "phenotype_cp_min", "hdf5"),
    ],
    "eval_segmentation_phenotype": [
        PHENOTYPE_PROCESS_FP / "eval" / "segmentation" / "segmentation_overview.tsv",
        PHENOTYPE_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.tsv",
        PHENOTYPE_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.png",
    ],
    # create heatmap tsv and png for each evaluated feature
    "eval_features": [
        PHENOTYPE_PROCESS_FP / "eval" / "features" / f"{feature}_heatmap.tsv"
        for feature in eval_features
    ]
    + [
        PHENOTYPE_PROCESS_FP / "eval" / "features" / f"{feature}_heatmap.png"
        for feature in eval_features
    ],
}

PHENOTYPE_PROCESS_OUTPUT_MAPPINGS = {
    "apply_ic_field_phenotype": None,
    "segment_phenotype": None,
    "identify_cytoplasm": None,
    "extract_phenotype_info": None,
    "merge_phenotype_info": None,
    "extract_phenotype_cp": None,
    "merge_phenotype_cp": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

PHENOTYPE_PROCESS_WILDCARDS = {
    "well": PHENOTYPE_WELLS,
    "tile": PHENOTYPE_TILES,
}

if config['phenotype_process']['mode'] == 'segment_phenotype_paramsearch':
    PHENOTYPE_PROCESS_OUTPUTS.update({
        "segment_phenotype_paramsearch": [
            PHENOTYPE_PROCESS_FP / "paramsearch" / "images" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_nuclei",
                "tiff"),
            PHENOTYPE_PROCESS_FP / "paramsearch" / "images" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_cells",
                "tiff"),
            PHENOTYPE_PROCESS_FP / "paramsearch" / "tsvs" / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_segmentation_stats",
                "tsv")
        ],
        "summarize_segment_phenotype_paramsearch": [
            PHENOTYPE_PROCESS_FP / "paramsearch" / "summary" / "segmentation_summary.tsv",  
            PHENOTYPE_PROCESS_FP / "paramsearch" / "summary" / "segmentation_grouped.tsv",  
            PHENOTYPE_PROCESS_FP / "paramsearch" / "summary" / "segmentation_evaluation.txt", 
            PHENOTYPE_PROCESS_FP / "paramsearch" / "summary" / "segmentation_panel.png",
        ]
    })
    
    PHENOTYPE_PROCESS_OUTPUT_MAPPINGS.update({
        "segment_phenotype_paramsearch": None,
        "summarize_segment_phenotype_paramsearch": None
    })
    
    PHENOTYPE_PROCESS_WILDCARDS.update({
        "nuclei_diameter": config["phenotype_process"]["paramsearch"]["nuclei_diameter"],
        "cell_diameter": config["phenotype_process"]["paramsearch"]["cell_diameter"],
        "flow_threshold": config["phenotype_process"]["paramsearch"]["flow_threshold"],
        "cellprob_threshold": config["phenotype_process"]["paramsearch"]["cellprob_threshold"]
    })


PHENOTYPE_PROCESS_OUTPUTS_MAPPED = map_outputs(
    PHENOTYPE_PROCESS_OUTPUTS, PHENOTYPE_PROCESS_OUTPUT_MAPPINGS
)

PHENOTYPE_PROCESS_TARGETS = outputs_to_targets(
    PHENOTYPE_PROCESS_OUTPUTS, PHENOTYPE_PROCESS_WILDCARDS
)

PHENOTYPE_PROCESS_TARGETS_ALL = sum(PHENOTYPE_PROCESS_TARGETS.values(), [])
