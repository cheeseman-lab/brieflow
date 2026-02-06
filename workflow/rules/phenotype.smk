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
    params:
        segment_cells=config["phenotype"].get("segment_cells", True),
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


# Extract full phenotype information from phenotype images
rule extract_phenotype:
    input:
        # aligned phenotype image
        PHENOTYPE_OUTPUTS["align_phenotype"][0],
        # nuclei segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][0],
        # cells segmentation map
        PHENOTYPE_OUTPUTS["segment_phenotype"][1],
        # cytoplasm segmentation map
        PHENOTYPE_OUTPUTS["identify_cytoplasm"][0],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["extract_phenotype"],
    params:
        foci_channel_index=config["phenotype"]["foci_channel_index"],
        channel_names=config["phenotype"]["channel_names"],
        cp_method=config["phenotype"]["cp_method"],
        segment_cells=config["phenotype"].get("segment_cells", True),
    script:
        "../scripts/phenotype/extract_phenotype.py"


# Combine phenotype results from different tiles
rule merge_phenotype:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["extract_phenotype"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    params:
        channel_names=config["phenotype"]["channel_names"],
        segment_cells=config["phenotype"].get("segment_cells", True),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["merge_phenotype"],
    script:
        "../scripts/phenotype/merge_phenotype.py"


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
        heatmap_shape=config["phenotype"].get("heatmap_shape", "6W_ph"),
        heatmap_plate=config["phenotype"].get("heatmap_plate", "6W"),
    script:
        "../scripts/shared/eval_segmentation.py"


rule eval_features:
    input:
        # use minimum phenotype features for evaluation
        cells_paths=lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["merge_phenotype"][1],
            wildcards=wildcards,
            expansion_values=["well", "tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["eval_features"],
    params:
        heatmap_shape=config["phenotype"].get("heatmap_shape", "6W_ph"),
        heatmap_plate=config["phenotype"].get("heatmap_plate", "6W"),
    script:
        "../scripts/phenotype/eval_features.py"


# Assemble HCS plate-level zarr stores from per-tile outputs (zarr mode only)
if PHENOTYPE_IMG_FMT == "zarr":
    rule finalize_hcs_phenotype:
        input:
            PHENOTYPE_TARGETS_ALL,
        output:
            touch(str(PHENOTYPE_FP / ".hcs_done")),
        params:
            images_dir=str(PHENOTYPE_FP / "images"),
            hcs_dir=str(PHENOTYPE_FP / "hcs"),
        script:
            "../scripts/shared/write_hcs_metadata.py"


# Rule for all phenotype processing steps
rule all_phenotype:
    input:
        PHENOTYPE_TARGETS_ALL,
