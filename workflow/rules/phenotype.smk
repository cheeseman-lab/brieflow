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
rule merge_phenotype_info:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["extract_phenotype_info"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["merge_phenotype_info"],
    script:
        "../scripts/shared/combine_dfs.py"


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
        foci_channel=config["phenotype"]["foci_channel"],
        channel_names=config["phenotype"]["channel_names"],
    script:
        "../scripts/phenotype/extract_phenotype_cp_multichannel.py"


# Combine phenotype results from different tiles
rule merge_phenotype_cp:
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_OUTPUTS["extract_phenotype_cp"],
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
            PHENOTYPE_OUTPUTS["merge_phenotype_info"][0],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PHENOTYPE_OUTPUTS_MAPPED["eval_segmentation_phenotype"],
    params:
        heatmap_shape="6W_ph"
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


# TODO: test and implement segmentation paramsearch for updated brieflow setup
# if config["phenotype"]["mode"] == "segment_phenotype_paramsearch":
#     rule segment_phenotype_paramsearch:
#         input:
#             PHENOTYPE_OUTPUTS["align_phenotype"],
#         output:
#             PHENOTYPE_OUTPUTS_MAPPED["segment_phenotype_paramsearch"],
#         params:
#             dapi_index=config["phenotype"]["dapi_index"],
#             cyto_index=config["phenotype"]["cyto_index"],
#             nuclei_diameter=lambda wildcards: float(wildcards.nuclei_diameter),
#             cell_diameter=lambda wildcards: float(wildcards.cell_diameter),
#             cyto_model=config["phenotype"]["cyto_model"],
#             flow_threshold=lambda wildcards: float(wildcards.flow_threshold),
#             cellprob_threshold=lambda wildcards: float(wildcards.cellprob_threshold),
#             return_counts=True,
#             gpu=config["phenotype"]["gpu"],
#         script:
#             "../scripts/shared/segment_cellpose.py"

#     rule summarize_segment_phenotype_paramsearch:
#         input:
#             lambda wildcards: output_to_input(
#                 PHENOTYPE_OUTPUTS["segment_phenotype_paramsearch"][2::3],
#                 {
#                     "well": PHENOTYPE_WELLS,
#                     "tile": PHENOTYPE_TILES,
#                     "nuclei_diameter": PHENOTYPE_WILDCARDS["nuclei_diameter"],
#                     "cell_diameter": PHENOTYPE_WILDCARDS["cell_diameter"],
#                     "flow_threshold": PHENOTYPE_WILDCARDS["flow_threshold"],
#                     "cellprob_threshold": PHENOTYPE_WILDCARDS["cellprob_threshold"],
#                 },
#                 wildcards,
#             ),
#         output:
#             PHENOTYPE_OUTPUTS_MAPPED["summarize_segment_phenotype_paramsearch"],
#         params:
#             segmentation="phenotype",
#             channel_cmaps=config["phenotype"]["channel_cmaps"],
#             cell_diameter=config["phenotype"]["cell_diameter"],
#             nuclei_diameter=config["phenotype"]["nuclei_diameter"],
#             cellprob_threshold=config["phenotype"]["cellprob_threshold"],
#             flow_threshold=config["phenotype"]["flow_threshold"],
#             output_type="tsv",
#         script:
#             "../scripts/shared/eval_segmentation_paramsearch.py"


# Rule for all phenotype processing steps
rule all_phenotype:
    input:
        PHENOTYPE_TARGETS_ALL,
