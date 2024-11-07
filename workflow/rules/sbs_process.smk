from pathlib import Path
from lib.shared.file_utils import get_filename

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]
SBS_PROCESS_FP = ROOT_FP / config["sbs_process"]["suffix"]


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
            cycle=SBS_CYCLES[config["sbs_process"]["segmentation_cycle"]],
        ),
    output:
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    params:
        segmentation_cycle=SBS_CYCLES[config["sbs_process"]["segmentation_cycle"]],
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
    params:
        dapi_index=config["sbs_process"]["dapi_index"],
        cyto_index=config["sbs_process"]["cyto_index"],
        nuclei_diameter=config["sbs_process"]["nuclei_diameter"],
        cell_diameter=config["sbs_process"]["cell_diameter"],
        cyto_model=config["sbs_process"]["cyto_model"],
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


# Extract minimal phenotype features
rule extract_phenotype_minimal:
    conda:
        "../envs/sbs_process.yml"
    input:
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"),
    output:
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "minimal_phenotype_info", "tsv"
        ),
    script:
        "../scripts/sbs_process/extract_phenotype_minimal.py"


# Rule for combining read results from different wells
rule combine_reads:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename({"well": wildcards.well, "tile": "{tile}"}, "reads", "tsv"),
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP / "hdfs" / get_filename({"well": "{well}"}, "reads", "hdf5"),
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
            / get_filename({"well": wildcards.well, "tile": "{tile}"}, "cells", "tsv"),
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP / "hdfs" / get_filename({"well": "{well}"}, "cells", "hdf5"),
    script:
        "../scripts/shared/combine_dfs.py"


# Rule for combining phenotypic info results from different wells
rule combine_minimal_phenotype_info:
    conda:
        "../envs/sbs_process.yml"
    input:
        lambda wildcards: expand(
            SBS_PROCESS_FP
            / "tsvs"
            / get_filename(
                {"well": wildcards.well, "tile": "{tile}"},
                "minimal_phenotype_info",
                "tsv",
            ),
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESS_FP
        / "hdfs"
        / get_filename({"well": "{well}"}, "minimal_phenotype_info", "hdf5"),
    script:
        "../scripts/shared/combine_dfs.py"
