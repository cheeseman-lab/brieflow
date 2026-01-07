from lib.shared.target_utils import output_to_input
from lib.shared.rule_utils import get_spot_detection_params, get_segmentation_params, get_call_cells_params


# Align images from each sequencing round
rule align_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_sbs"],
            wildcards=wildcards,
            expansion_values=["cycle"],
            metadata_combos=sbs_wildcard_combos,
            ancient_output=True,
        ),
    output:
        SBS_OUTPUTS_MAPPED["align_sbs"],
    params:
        method=config["sbs"]["alignment_method"],
        channel_names=config["sbs"]["channel_names"],
        upsample_factor=config["sbs"]["upsample_factor"],
        skip_cycles_indices=config["sbs"]["skip_cycles_indices"],
        manual_background_cycle_index=config["sbs"]["manual_background_cycle_index"],
        manual_channel_mapping=config["sbs"]["manual_channel_mapping"],
    script:
        "../scripts/sbs/align_cycles.py"


# Apply Laplacian-of-Gaussian filter to all channels
rule log_filter:
    input:
        SBS_OUTPUTS["align_sbs"],
    output:
        SBS_OUTPUTS_MAPPED["log_filter"],
    params:
        skip_index=config["sbs"]["extra_channel_indices"],
    script:
        "../scripts/sbs/log_filter.py"


# Compute standard deviation of SBS reads across cycles
rule compute_standard_deviation:
    input:
        SBS_OUTPUTS["log_filter"],
    output:
        SBS_OUTPUTS_MAPPED["compute_standard_deviation"],
    params:
        remove_index=config["sbs"]["extra_channel_indices"],
    script:
        "../scripts/sbs/compute_standard_deviation.py"


# Find local maxima of SBS reads across cycles
rule find_peaks:
    input:
        SBS_OUTPUTS["compute_standard_deviation"] if config["sbs"]["spot_detection_method"] == "standard" else SBS_OUTPUTS["align_sbs"],
    output:
        SBS_OUTPUTS_MAPPED["find_peaks"],
    params:
        config=lambda wildcards: get_spot_detection_params(config)
    script:
        "../scripts/sbs/find_peaks.py"


# Dilate sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    input:
        SBS_OUTPUTS["log_filter"],
    output:
        SBS_OUTPUTS_MAPPED["max_filter"],
    params:
        width=config["sbs"]["max_filter_width"],
        remove_index=config["sbs"]["extra_channel_indices"],
    script:
        "../scripts/sbs/max_filter.py"


# Apply illumination correction field from segmentation cycle
rule apply_ic_field_sbs:
    input:
        SBS_OUTPUTS["align_sbs"],
        # extra channel illumination correction field
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["calculate_ic_sbs"],
            wildcards=wildcards,
            subset_values={
                "cycle": str(config["sbs"]["dapi_cycle"])
            },
            ancient_output=True,
        ),
        # illumination correction field from cycle of interest
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["calculate_ic_sbs"],
            wildcards=wildcards,
            subset_values={
                "cycle": str(config["sbs"]["cyto_cycle"]),
            },
            ancient_output=True,
        ),
    output:
        SBS_OUTPUTS_MAPPED["apply_ic_field_sbs"],
    params:
        dapi_cycle=config["sbs"]["dapi_cycle"],
        cyto_cycle=config["sbs"]["cyto_cycle"],
        cyto_cycle_index=config["sbs"]["cyto_cycle_index"],
        extra_channel_indices=config["sbs"]["extra_channel_indices"],
    script:
        "../scripts/sbs/apply_ic_field_sbs.py"


# Segments cells and nuclei using pre-defined methods
rule segment_sbs:
    input:
        SBS_OUTPUTS["apply_ic_field_sbs"],
    output:
        SBS_OUTPUTS_MAPPED["segment_sbs"],
    params:
        config=lambda wildcards: get_segmentation_params("sbs", config),
    script:
        "../scripts/shared/segment.py"


# Extract bases from peaks
rule extract_bases:
    input:
        SBS_OUTPUTS["find_peaks"],
        SBS_OUTPUTS["max_filter"],
        # optionally use cell or nuclei segmentation
        lambda wildcards: SBS_OUTPUTS["segment_sbs"][1] if config["sbs"]["segment_cells"] else SBS_OUTPUTS["segment_sbs"][0],
    output:
        SBS_OUTPUTS_MAPPED["extract_bases"],
    params:
        threshold_peaks=config["sbs"]["threshold_peaks"],
        bases=config["sbs"]["bases"],
    script:
        "../scripts/sbs/extract_bases.py"


# Call reads
rule call_reads:
    input:
        SBS_OUTPUTS["extract_bases"],
        SBS_OUTPUTS["find_peaks"],
    output:
        SBS_OUTPUTS_MAPPED["call_reads"],
    params:
        call_reads_method=config["sbs"]["call_reads_method"]
    script:
        "../scripts/sbs/call_reads.py"


# Call cells (supports both single and multi-barcode protocols)
rule call_cells:
    input:
        SBS_OUTPUTS["call_reads"],
    output:
        SBS_OUTPUTS_MAPPED["call_cells"],
    params:
        config=lambda wildcards: get_call_cells_params(config),
    script:
        "../scripts/sbs/call_cells.py"


# Extract minimal sbs info
rule extract_sbs_info:
    input:
        # use nuclei segmentation map
        SBS_OUTPUTS["segment_sbs"][0],
    output:
        SBS_OUTPUTS_MAPPED["extract_sbs_info"],
    script:
        "../scripts/shared/extract_phenotype_minimal.py"


# Rule for combining read results from different wells
rule combine_reads:
    input:
        lambda wildcards: output_to_input(
            SBS_OUTPUTS["call_reads"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        SBS_OUTPUTS_MAPPED["combine_reads"],
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining cell results from different wells
rule combine_cells:
    input:
        lambda wildcards: output_to_input(
            SBS_OUTPUTS["call_cells"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        SBS_OUTPUTS_MAPPED["combine_cells"],
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining sbs info results from different wells
rule combine_sbs_info:
    input:
        lambda wildcards: output_to_input(
            SBS_OUTPUTS["extract_sbs_info"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        SBS_OUTPUTS_MAPPED["combine_sbs_info"],
    script:
        "../scripts/shared/combine_dfs.py"


rule eval_segmentation_sbs:
    input:
        # path to segmentation stats for well/tile
        segmentation_stats_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["segment_sbs"][2],
            wildcards=wildcards,
            expansion_values=["well", "tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
        # path to combined cell data
        cells_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        SBS_OUTPUTS_MAPPED["eval_segmentation_sbs"],
    params:
        heatmap_plate=config["sbs"].get("heatmap_plate", "6W"),   
        heatmap_shape=config["sbs"].get("heatmap_shape", "6W_sbs")
    script:
        "../scripts/shared/eval_segmentation.py"


rule eval_mapping:
    input:
        reads_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_reads"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=sbs_wildcard_combos,
        ),
        cells_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_cells"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=sbs_wildcard_combos,
        ),
        sbs_info_paths=lambda wildcards: output_to_input(
            SBS_OUTPUTS["combine_sbs_info"],
            wildcards=wildcards,
            expansion_values=["well"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        SBS_OUTPUTS_MAPPED["eval_mapping"],
    params:
        df_barcode_library_fp=config["sbs"]["df_barcode_library_fp"],
        heatmap_plate=config["sbs"].get("heatmap_plate", "6W"),
        heatmap_shape=config["sbs"].get("heatmap_shape", "6W_sbs"),
        sort_by=config["sbs"]["sort_calls"],
        barcode_type=config["sbs"].get("barcode_type", "simple"),
        sequencing_order=config["sbs"].get("sequencing_order", "map_recomb"),
        library_barcode_col=(
            config["sbs"].get("map_col", "prefix_map")
            if config["sbs"].get("barcode_type", "simple") == "multi"
            else config["sbs"].get("prefix_col", "prefix")
        ),
        recomb_col=config["sbs"].get("recomb_col", "prefix_recomb"),
    script:
        "../scripts/sbs/eval_mapping.py"


# rule for all sbs processing steps
rule all_sbs:
    input:
        SBS_TARGETS_ALL,
