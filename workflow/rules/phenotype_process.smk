from lib.shared.target_utils import output_to_input


# Apply illumination correction field
rule apply_ic_field_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PREPROCESS_OUTPUTS["convert_phenotype"],
        PREPROCESS_OUTPUTS["calculate_ic_phenotype"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["apply_ic_field_phenotype"],
    script:
        "../scripts/phenotype_process/apply_ic_field_phenotype.py"

# Align phenotype images
rule align_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PHENOTYPE_PROCESS_OUTPUTS["apply_ic_field_phenotype"]
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["align_phenotype"]
    params:
        align=config["phenotype_process"]["align"],
        target=lambda wildcards: config["phenotype_process"]["target"] if config["phenotype_process"]["align"] else None,
        source=lambda wildcards: config["phenotype_process"]["source"] if config["phenotype_process"]["align"] else None,
        riders=lambda wildcards: config["phenotype_process"]["riders"] if config["phenotype_process"]["align"] else None,
        remove_channel=lambda wildcards: config["phenotype_process"]["remove_channel"] if config["phenotype_process"]["align"] else None,
    script:
        "../scripts/phenotype_process/align_phenotype.py"

# Segments cells and nuclei using pre-defined methods
rule segment_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PHENOTYPE_PROCESS_OUTPUTS["align_phenotype"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["segment_phenotype"],
    params:
        dapi_index=config["phenotype_process"]["dapi_index"],
        cyto_index=config["phenotype_process"]["cyto_index"],
        nuclei_diameter=config["phenotype_process"]["nuclei_diameter"],
        cell_diameter=config["phenotype_process"]["cell_diameter"],
        cyto_model=config["phenotype_process"]["cyto_model"],
        flow_threshold=config["phenotype_process"]["flow_threshold"],
        cellprob_threshold=config["phenotype_process"]["cellprob_threshold"],
        return_counts=True,
        gpu=config["phenotype_process"]["gpu"],
    script:
        "../scripts/shared/segment_cellpose.py"


# Extract cytoplasmic masks from segmented nuclei, cells
rule identify_cytoplasm:
    conda:
        "../envs/phenotype_process.yml"
    input:
        # nuclei segmentation map
        PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][0],
        # cells segmentation map
        PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][1],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["identify_cytoplasm"],
    script:
        "../scripts/phenotype_process/identify_cytoplasm_cellpose.py"


# Extract minimal phenotype information from segmented nuclei images
rule extract_phenotype_info:
    conda:
        "../envs/phenotype_process.yml"
    input:
        # nuclei segmentation map
        PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][0],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["extract_phenotype_info"],
    script:
        "../scripts/shared/extract_phenotype_minimal.py"


# Combine phenotype info results from different wells
rule merge_phenotype_info:
    conda:
        "../envs/phenotype_process.yml"
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_PROCESS_OUTPUTS["extract_phenotype_info"],
            {"tile": PHENOTYPE_TILES, "well": PHENOTYPE_WELLS},
            wildcards,
        ),
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["merge_phenotype_info"],
    script:
        "../scripts/shared/combine_dfs.py"


# Extract full phenotype information using CellProfiler from phenotype images
rule extract_phenotype_cp:
    conda:
        "../envs/phenotype_process.yml"
    input:
        PHENOTYPE_PROCESS_OUTPUTS["align_phenotype"],
        # nuclei segmentation map
        PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][0],
        # cells segmentation map
        PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][1],
        PHENOTYPE_PROCESS_OUTPUTS["identify_cytoplasm"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["extract_phenotype_cp"],
    params:
        foci_channel=config["phenotype_process"]["foci_channel"],
        channel_names=config["phenotype_process"]["channel_names"],
    script:
        "../scripts/phenotype_process/extract_phenotype_cp_multichannel.py"


# Combine phenotype results from different wells
rule merge_phenotype_cp:
    conda:
        "../envs/phenotype_process.yml"
    input:
        lambda wildcards: output_to_input(
            PHENOTYPE_PROCESS_OUTPUTS["extract_phenotype_cp"],
            {"tile": PHENOTYPE_TILES, "well": PHENOTYPE_WELLS},
            wildcards,
        ),
    params:
        channel_names=config["phenotype_process"]["channel_names"],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["merge_phenotype_cp"],
    script:
        "../scripts/phenotype_process/merge_phenotype_cp.py"


# Evaluate segmentation results
rule eval_segmentation_phenotype:
    conda:
        "../envs/phenotype_process.yml"
    input:
        # path to segmentation stats for well/tile
        segmentation_stats_paths=lambda wildcards: output_to_input(
            PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype"][2],
            {"well": PHENOTYPE_WELLS, "tile": PHENOTYPE_TILES},
            wildcards,
        ),
        # path to hdf with combined cell data
        cells_path=PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_info"][0],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["eval_segmentation_phenotype"],
    script:
        "../scripts/shared/eval_segmentation.py"


rule eval_features:
    conda:
        "../envs/phenotype_process.yml"
    input:
        # use minimum phenotype CellProfiler features for evaluation
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_cp"][1],
    output:
        PHENOTYPE_PROCESS_OUTPUTS_MAPPED["eval_features"],
    script:
        "../scripts/phenotype_process/eval_features.py"


if config['phenotype_process']['mode'] == 'segment_phenotype_paramsearch':
    rule segment_phenotype_paramsearch:
        conda:
            "../envs/phenotype_process.yml"
        input:
            PHENOTYPE_PROCESS_OUTPUTS["align_phenotype"]
        output:
            PHENOTYPE_PROCESS_OUTPUTS_MAPPED["segment_phenotype_paramsearch"]
        params:
            dapi_index=config["phenotype_process"]["dapi_index"],
            cyto_index=config["phenotype_process"]["cyto_index"],
            nuclei_diameter=lambda wildcards: float(wildcards.nuclei_diameter),
            cell_diameter=lambda wildcards: float(wildcards.cell_diameter),
            cyto_model=config["phenotype_process"]["cyto_model"],
            flow_threshold=lambda wildcards: float(wildcards.flow_threshold),
            cellprob_threshold=lambda wildcards: float(wildcards.cellprob_threshold),
            return_counts=True,
            gpu=config["phenotype_process"]["gpu"],
        script:
            "../scripts/shared/segment_cellpose.py"

    rule summarize_segment_phenotype_paramsearch:
        conda:
            "../envs/phenotype_process.yml"
        input:
            lambda wildcards: output_to_input(
                PHENOTYPE_PROCESS_OUTPUTS["segment_phenotype_paramsearch"][2::3],
                {"well": PHENOTYPE_WELLS, "tile": PHENOTYPE_TILES,
                "nuclei_diameter": PHENOTYPE_PROCESS_WILDCARDS["nuclei_diameter"],
                "cell_diameter": PHENOTYPE_PROCESS_WILDCARDS["cell_diameter"],
                "flow_threshold": PHENOTYPE_PROCESS_WILDCARDS["flow_threshold"],
                "cellprob_threshold": PHENOTYPE_PROCESS_WILDCARDS["cellprob_threshold"]},
                wildcards,
            )
        output:
            PHENOTYPE_PROCESS_OUTPUTS_MAPPED["summarize_segment_phenotype_paramsearch"]
        params:
            segmentation_process="phenotype_process",
            channel_cmaps=config["phenotype_process"]["channel_cmaps"],
            cell_diameter=config["phenotype_process"]["cell_diameter"],
            nuclei_diameter=config["phenotype_process"]["nuclei_diameter"],
            cellprob_threshold=config["phenotype_process"]["cellprob_threshold"],
            flow_threshold=config["phenotype_process"]["flow_threshold"],
            output_type="tsv"
        script:
            "../scripts/shared/eval_segmentation_paramsearch.py"


# Rule for all phenotype processing steps
rule all_phenotype_process:
    input:
        PHENOTYPE_PROCESS_TARGETS_ALL,
