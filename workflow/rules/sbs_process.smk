from lib.shared.target_utils import output_to_input


# Align images from each sequencing round
rule align:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_sbs"],
            {"cycle": SBS_CYCLES},
            wildcards,
        ),
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["align"],
    params:
        method="sbs_mean",
        upsample_factor=1,
    script:
        "../scripts/sbs_process/align_cycles.py"


# Apply Laplacian-of-Gaussian filter to all channels
rule log_filter:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["align"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["log_filter"],
    params:
        skip_index=0,
    script:
        "../scripts/sbs_process/log_filter.py"


# Compute standard deviation of SBS reads across cycles
rule compute_standard_deviation:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["log_filter"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["compute_standard_deviation"],
    params:
        remove_index=0,
    script:
        "../scripts/sbs_process/compute_standard_deviation.py"


# Find local maxima of SBS reads across cycles
rule find_peaks:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["compute_standard_deviation"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["find_peaks"],
    script:
        "../scripts/sbs_process/find_peaks.py"


# Dilate sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["log_filter"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["max_filter"],
    params:
        width=3,
        remove_index=0,
    script:
        "../scripts/sbs_process/max_filter.py"


# Apply illumination correction field from segmentation cycle
rule apply_ic_field_sbs:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["align"],
        # illumination correction field from cycle of interest
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["calculate_ic_sbs"],
            {"cycle": SBS_CYCLES[config["sbs_process"]["segmentation_cycle_index"]]},
            wildcards,
        ),
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["apply_ic_field_sbs"],
    params:
        segmentation_cycle_index=SBS_CYCLES[
            config["sbs_process"]["segmentation_cycle_index"]
        ],
    script:
        "../scripts/sbs_process/apply_ic_field_sbs.py"


# Segments cells and nuclei using pre-defined methods
rule segment_sbs:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["apply_ic_field_sbs"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["segment_sbs"],
    params:
        dapi_index=config["sbs_process"]["dapi_index"],
        cyto_index=config["sbs_process"]["cyto_index"],
        nuclei_diameter=config["sbs_process"]["nuclei_diameter"],
        cell_diameter=config["sbs_process"]["cell_diameter"],
        cyto_model=config["sbs_process"]["cyto_model"],
        flow_threshold=config["sbs_process"]["flow_threshold"],
        cellprob_threshold=config["sbs_process"]["cellprob_threshold"],
        return_counts=True,
        gpu=config["sbs_process"]["gpu"],
    script:
        "../scripts/shared/segment_cellpose.py"


# Extract bases from peaks
rule extract_bases:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["find_peaks"],
        SBS_PROCESS_OUTPUTS["max_filter"],
        # use cell segmentation map
        SBS_PROCESS_OUTPUTS["segment_sbs"][1],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["extract_bases"],
    params:
        threshold_peaks=config["sbs_process"]["threshold_peaks"],
        bases=config["sbs_process"]["bases"],
    script:
        "../scripts/sbs_process/extract_bases.py"


# Call reads
rule call_reads:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["extract_bases"],
        SBS_PROCESS_OUTPUTS["find_peaks"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["call_reads"],
    script:
        "../scripts/sbs_process/call_reads.py"


# Call cells
rule call_cells:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["call_reads"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["call_cells"],
    params:
        df_design_path=config["sbs_process"]["df_design_path"],
        q_min=config["sbs_process"]["q_min"],
    script:
        "../scripts/sbs_process/call_cells.py"


# Extract minimal sbs info
rule extract_sbs_info:
    conda:
        "../envs/sbs_process.yml"
    input:
        # use nuclei segmentation map
        SBS_PROCESS_OUTPUTS["segment_sbs"][0],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["extract_sbs_info"],
    script:
        "../scripts/shared/extract_phenotype_minimal.py"


# Rule for combining read results from different wells
rule combine_reads:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: output_to_input(
            SBS_PROCESS_OUTPUTS["call_reads"],
            {"well": SBS_WELLS, "tile": SBS_TILES},
            wildcards,
        ),
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["combine_reads"],
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining cell results from different wells
rule combine_cells:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: output_to_input(
            SBS_PROCESS_OUTPUTS["call_cells"],
            {"well": SBS_WELLS, "tile": SBS_TILES},
            wildcards,
        ),
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["combine_cells"],
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining sbs info results from different wells
rule combine_sbs_info:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: output_to_input(
            SBS_PROCESS_OUTPUTS["extract_sbs_info"],
            {"well": SBS_WELLS, "tile": SBS_TILES},
            wildcards,
        ),
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["combine_sbs_info"],
    script:
        "../scripts/shared/combine_dfs.py"


rule eval_segmentation_sbs:
    conda:
        "../envs/sbs_process.yml"
    input:
        # path to segmentation stats for well/tile
        segmentation_stats_paths=lambda wildcards: output_to_input(
            SBS_PROCESS_OUTPUTS["segment_sbs"][2],
            {"well": SBS_WELLS, "tile": SBS_TILES},
            wildcards,
        ),
        # path to hdf with combined cell data
        cells_path=SBS_PROCESS_OUTPUTS["combine_cells"][0],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["eval_segmentation_sbs"],
    script:
        "../scripts/shared/eval_segmentation.py"


rule eval_mapping:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_OUTPUTS["combine_reads"],
        SBS_PROCESS_OUTPUTS["combine_cells"],
        SBS_PROCESS_OUTPUTS["combine_sbs_info"],
    output:
        SBS_PROCESS_OUTPUTS_MAPPED["eval_mapping"],
    params:
        df_design_path=config["sbs_process"]["df_design_path"],
    script:
        "../scripts/sbs_process/eval_mapping.py"


if config['sbs_process']['mode'] == 'segment_sbs_paramsearch':
    rule segment_sbs_paramsearch:
        conda:
            "../envs/sbs_process.yml"
        input:
            SBS_PROCESS_OUTPUTS["apply_ic_field_sbs"],
        output:
            SBS_PROCESS_OUTPUTS_MAPPED["segment_sbs_paramsearch"]
        params:
            dapi_index=config["sbs_process"]["dapi_index"],
            cyto_index=config["sbs_process"]["cyto_index"], 
            nuclei_diameter=lambda wildcards: float(wildcards.nuclei_diameter),
            cell_diameter=lambda wildcards: float(wildcards.cell_diameter),
            cyto_model=config["sbs_process"]["cyto_model"],
            flow_threshold=lambda wildcards: float(wildcards.flow_threshold),
            cellprob_threshold=lambda wildcards: float(wildcards.cellprob_threshold),
            return_counts=True,
            gpu=config["sbs_process"]["gpu"],

        script:
            "../scripts/shared/segment_cellpose.py"

    rule summarize_segment_sbs_paramsearch:
        conda:
            "../envs/sbs_process.yml" 
        input:
            lambda wildcards: output_to_input(
                SBS_PROCESS_OUTPUTS["segment_sbs_paramsearch"][2::3],
                {"well": SBS_WELLS, "tile": SBS_TILES,
                    "nuclei_diameter": SBS_PROCESS_WILDCARDS["nuclei_diameter"],
                    "cell_diameter": SBS_PROCESS_WILDCARDS["cell_diameter"],
                    "flow_threshold": SBS_PROCESS_WILDCARDS["flow_threshold"],
                    "cellprob_threshold": SBS_PROCESS_WILDCARDS["cellprob_threshold"]},
                wildcards,
            )
        output:
            SBS_PROCESS_OUTPUTS_MAPPED["summarize_segment_sbs_paramsearch"]
        params:
            segmentation_process="sbs_process",
            dapi_index=config["sbs_process"]["dapi_index"],
            cyto_index=config["sbs_process"]["cyto_index"],
            cell_diameter=config["sbs_process"]["cell_diameter"],
            nuclei_diameter=config["sbs_process"]["nuclei_diameter"],
            cellprob_threshold=config["sbs_process"]["cellprob_threshold"],
            flow_threshold=config["sbs_process"]["flow_threshold"],
            output_type="tsv"
        script:
            "../scripts/shared/eval_segmentation_paramsearch.py"


# rule for all sbs processing steps
rule all_sbs_process:
    input:
        SBS_PROCESS_TARGETS_ALL,
