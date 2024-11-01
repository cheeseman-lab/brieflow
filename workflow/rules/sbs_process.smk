from pathlib import Path
from lib.shared.file_utils import get_filename

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]
SBS_PROCESS_FP = ROOT_FP / config["sbs_process"]["suffix"]


# Aligns images from each sequencing round
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


# Applies Laplacian-of-Gaussian filter to all channels
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


# Computes standard deviation of SBS reads across cycles
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
