from lib.shared.file_utils import get_filename

PREPROCESS_FP = ROOT_FP / "preprocess"
SBS_PROCESS_FP = ROOT_FP / "sbs_process"


# Align images from each sequencing round
rule align:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            PREPROCESS_FP
            / "images"
            / "sbs"
            / get_filename(
                {"well": wildcards.well, "tile": wildcards.tile, "cycle": "{cycle}"},
                "image",
                "tiff",
            ),
            cycle=SBS_CYCLES,
        ),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tiff"),
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
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tiff"),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "log_filtered", "tiff"),
    params:
        skip_index=0,
    script:
        "../scripts/sbs_process/log_filter.py"


# Compute standard deviation of SBS reads across cycles
rule compute_standard_deviation:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "log_filtered", "tiff"),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "standard_deviation", "tiff"
        ),
    params:
        remove_index=0,
    script:
        "../scripts/sbs_process/compute_standard_deviation.py"


# Find local maxima of SBS reads across cycles
rule find_peaks:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "standard_deviation", "tiff"
        ),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "peaks", "tiff"),
    script:
        "../scripts/sbs_process/find_peaks.py"


# Dilate sequencing channels to compensate for single-pixel alignment error.
rule max_filter:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "log_filtered", "tiff"),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "max_filtered", "tiff"),
    params:
        width=3,
        remove_index=0,
    script:
        "../scripts/sbs_process/max_filter.py"


# Apply illumination correction field from segmentation cycle
rule apply_ic_field:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tiff"),
        # illumination correction field from cycle of interest
        lambda wildcards: expand(
            PREPROCESS_FP
            / "ic_fields"
            / "sbs"
            / get_filename(
                {"well": wildcards.well, "cycle": "{cycle}"},
                "ic_field",
                "tiff",
            ),
            cycle=SBS_CYCLES[config["sbs_process"]["segmentation_cycle_index"]],
        ),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    params:
        segmentation_cycle_index=SBS_CYCLES[
            config["sbs_process"]["segmentation_cycle_index"]
        ],
    script:
        "../scripts/sbs_process/apply_ic_field.py"


# Segments cells and nuclei using pre-defined methods
rule segment:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"),
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tiff"),
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "segmentation_stats", "tsv"
        ),
    params:
        dapi_index=config["sbs_process"]["dapi_index"],
        cyto_index=config["sbs_process"]["cyto_index"],
        nuclei_diameter=config["sbs_process"]["nuclei_diameter"],
        cell_diameter=config["sbs_process"]["cell_diameter"],
        cyto_model=config["sbs_process"]["cyto_model"],
        return_counts=True,
    script:
        "../scripts/sbs_process/segment_cellpose.py"


# Extract bases from peaks
rule extract_bases:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "peaks", "tiff"),
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "max_filtered", "tiff"),
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tiff"),
    output:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "bases", "tsv"),
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
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "bases", "tsv"),
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "peaks", "tiff"),
    output:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "reads", "tsv"),
    script:
        "../scripts/sbs_process/call_reads.py"


# Call cells
rule call_cells:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "reads", "tsv"),
    output:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tsv"),
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
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"),
    output:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "sbs_info", "tsv"),
    script:
        "../scripts/sbs_process/extract_sbs_info.py"


# Rule for combining read results from different wells
rule combine_reads:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename({"well": "{well}", "tile": "{tile}"}, "reads", "tsv"),
            well=SBS_WELLS,
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "reads", "hdf5"),
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining cell results from different wells
rule combine_cells:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tsv"),
            well=SBS_WELLS,
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "cells", "hdf5"),
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining sbs info results from different wells
rule combine_sbs_info:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename(
                {"well": "{well}", "tile": "{tile}"},
                "sbs_info",
                "tsv",
            ),
            well=SBS_WELLS,
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "sbs_info", "hdf5"),
    script:
        "../scripts/shared/combine_dfs.py"


rule eval_segmentation:
    conda:
        "../envs/sbs_process.yml"
    input:
        segmentation_stats_paths=lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename(
                {"well": "{well}", "tile": "{tile}"}, "segmentation_stats", "tsv"
            ),
            well=SBS_WELLS,
            tile=SBS_TILES,
        ),
        cells_path=SBS_PROCESS_FP / "hdfs" / get_filename({}, "cells", "hdf5"),
    output:
        SBS_PROCESS_FP / "eval" / "segmentation" / "segmentation_overview.tsv",
        SBS_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.tsv",
        SBS_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.png",
    script:
        "../scripts/sbs_process/eval_segmentation.py"


rule eval_mapping:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "reads", "hdf5"),
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "cells", "hdf5"),
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "sbs_info", "hdf5"),
    output:
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_vs_threshold_peak.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_vs_threshold_qmin.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "read_mapping_heatmap.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_one.tsv",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_one.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_any.tsv",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_any.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "reads_per_cell_histogram.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "gene_symbol_histogram.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_overview.tsv",
    params:
        df_design_path=config["sbs_process"]["df_design_path"],
    script:
        "../scripts/sbs_process/eval_mapping.py"
