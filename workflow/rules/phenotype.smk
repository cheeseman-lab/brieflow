from lib.shared.target_utils import output_to_input
from lib.shared.rule_utils import get_alignment_params, get_segmentation_params


# Apply illumination correction field
rule apply_ic_field_phenotype:
    input:
        ancient(PREPROCESS_OUTPUTS["convert_phenotype"]),
        ancient(PREPROCESS_OUTPUTS["calculate_ic_phenotype"]),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["apply_ic_field_phenotype"],
    script:
        "../scripts/phenotype/apply_ic_field_phenotype.py"


# Align phenotype images
rule align_phenotype:
    input:
        PHENOTYPE_OUTPUTS["apply_ic_field_phenotype"],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["align_phenotype"],
    params:
        config=lambda wildcards: get_alignment_params(wildcards, config),
    script:
        "../scripts/phenotype/align_phenotype.py"


# Segments cells and nuclei using pre-defined methods
rule segment_phenotype:
    input:
        PHENOTYPE_OUTPUTS["align_phenotype"],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["segment_phenotype"],
    params:
        config=lambda wildcards: get_segmentation_params("phenotype", config),
    script:
        "../scripts/shared/segment.py"


# Extract cytoplasmic masks from segmented nuclei, cells
rule identify_cytoplasm:
    input:
        # nuclei segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][0],
        # cells segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][1],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["identify_cytoplasm"],
    script:
        "../scripts/phenotype/identify_cytoplasm_cellpose.py"


# Extract minimal phenotype information from segmented nuclei images
rule extract_phenotype_info:
    input:
        # nuclei segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][0],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["extract_phenotype_info"],
    script:
        "../scripts/shared/extract_phenotype_minimal.py"


# Combine phenotype info results from different tiles
rule combine_phenotype_info:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["extract_phenotype_info"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["combine_phenotype_info"],
    script:
        "../scripts/shared/combine_dfs.py"


# Identify secondary objects from aligned phenotype image and cell segmentation
if config["phenotype"].get("second_obj_detection", True):
    rule identify_second_objs:
        input:
            # aligned phenotype image
            PHENOTYPE_OUTPUTS["align_phenotype"],
            # cell segmentation map
            PHENOTYPE_OUTPUTS["segment_phenotype"][1],
            # cytoplasm mask
            PHENOTYPE_OUTPUTS["identify_cytoplasm"],
            # phenotype info with nuclei centroids
            PHENOTYPE_OUTPUTS["extract_phenotype_info"],
        output:
            # secondary object mask
            PHENOTYPE_OUTPUTS_MAPPED["identify_second_objs"][0],
            # cell secondary object table
            PHENOTYPE_OUTPUTS_MAPPED["identify_second_objs"][1],
            # updated cytoplasm masks
            PHENOTYPE_OUTPUTS_MAPPED["identify_second_objs"][2],
        params:
            # Pass all secondary object parameters from config
            second_obj_params=config["phenotype"],
        script:
            "../scripts/phenotype/identify_second_objs.py"

# Extract secondary object phenotype features
if config["phenotype"].get("second_obj_detection", True):
    rule extract_phenotype_second_objs:
        input:
            # aligned phenotype image
            PHENOTYPE_OUTPUTS["align_phenotype"],
            # secondary object mask
            PHENOTYPE_OUTPUTS["identify_second_objs"][0],
            # cell secondary object table
            PHENOTYPE_OUTPUTS["identify_second_objs"][1],
        output:
            PHENOTYPE_OUTPUTS_MAPPED["extract_phenotype_second_objs"],
        params:
            foci_channel_index=config["phenotype"]["foci_channel_index"],
            channel_names=config["phenotype"]["channel_names"],
        script:
            "../scripts/phenotype/extract_phenotype_second_objs.py"


# Combine secondary object phenotype results from different tiles
if config["phenotype"].get("second_obj_detection", True):
    rule merge_phenotype_second_objs:
        input:
            lambda wildcards: output_to_input(
                PHENOTYPE_OUTPUTS["extract_phenotype_second_objs"],
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=phenotype_wildcard_combos,
            ),
        params:
            channel_names=config["phenotype"]["channel_names"],
        output:
            PHENOTYPE_OUTPUTS_MAPPED["merge_phenotype_second_objs"],
        script:
            "../scripts/phenotype/merge_phenotype_second_objs.py"


# Extract full phenotype information using CellProfiler from phenotype images
rule extract_phenotype_cp:
    input:
        # aligned phenotype image
        PHENOTYPE_OUTPUTS["align_phenotype"],
        # nuclei segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][0],
        # cells segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][1],
        PHENOTYPE_OUTPUTS["identify_cytoplasm"],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["extract_phenotype_cp"],
    params:
        foci_channel_index=config["phenotype"]["foci_channel_index"],
        channel_names=config["phenotype"]["channel_names"],
        cp_method=config["phenotype"]["cp_method"],
    script:
        "../scripts/phenotype/extract_phenotype_cp_multichannel.py"


# Merge secondary object data with main phenotype data
if config["phenotype"].get("second_obj_detection", True):
    rule merge_second_objs_phenotype_cp:
        input:
            # main phenotype data 
            PHENOTYPE_OUTPUTS["extract_phenotype_cp"],
            # secondary object data 
            PHENOTYPE_OUTPUTS["identify_second_objs"][1],
        output:
            PHENOTYPE_OUTPUTS_MAPPED["merge_second_objs_phenotype_cp"],
        script:
            "../scripts/phenotype/merge_second_objs_phenotype_cp.py"


# Combine phenotype results from different tiles
rule merge_phenotype_cp:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_second_objs_phenotype_cp"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    params:
        channel_names=config["phenotype"]["channel_names"],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["merge_phenotype_cp"],
    script:
        "../scripts/phenotype/merge_phenotype_cp.py"


# Evaluate segmentation results
rule eval_segmentation_phenotype:
    input:
        # path to segmentation stats for well/tile
        segmentation_stats_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["segment_phenotype"][2],
            wildcards=wildcards,
            expansion_values=["well", "tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
        # paths to combined cell data
        cells_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["combine_phenotype_info"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["eval_segmentation_phenotype"],
    params:
        heatmap_shape="6W_ph",
    script:
        "../scripts/shared/eval_segmentation.py"


rule eval_features:
    input:
        # use minimum phenotype CellProfiler features for evaluation
        cells_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["eval_features"],
    script:
        "../scripts/phenotype/eval_features.py"


# Rule for all phenotype processing steps
rule all_phenotype:
    input:
        PHENOTYPE_TARGETS_ALL,
