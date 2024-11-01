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
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tsv"),
    params:
        method="sbs_mean",
        upsample_factor=1,
        display_ranges=config["sbs_process"]["display_ranges"],
    script:
        "../scripts/sbs_process/align_cycles.py"
