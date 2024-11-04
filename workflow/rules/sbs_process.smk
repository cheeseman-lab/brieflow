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
            / "images"
            / "sbs"
            / get_filename(
                {"well": wildcards.well, "tile": wildcards.tile, "cycle": "{cycle}"},
                "image",
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
