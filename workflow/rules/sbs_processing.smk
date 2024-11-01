from pathlib import Path
from lib.shared.file_utils import get_filename

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]
SBS_PROCESSING_FP = ROOT_FP / config["sbs_processing"]["suffix"]


# Aligns images from each sequencing round
rule align:
    input:
        [
            f"{INPUT_DIR}/sbs_tifs/{PREPROCESS_PATTERN}".format(
                cycle=cycle, well="{well}", tile="{tile}"
            )
            for cycle in SBS_CYCLES
        ],
    output:
        temp(f"{IMAGES_DIR}/10X_{{well}}_Tile-{{tile}}.aligned.tif"),
    run:
        # Read each cycle image into a list
        data = [read(f) for f in input]

        # Print number of data points for verification
        print(f"Number of images loaded: {len(data)}")

        # Call the alignment function from Snake_sbs
        Snake_sbs.align_SBS(
            output=output,
            data=data,
            method="SBS_mean",
            cycle_files=CYCLE_FILES,
            upsample_factor=1,
            n=1,
            keep_extras=False,
            display_ranges=DISPLAY_RANGES,
            luts=LUTS,
        )


# Aligns images from each sequencing round
rule align:
    conda:
        "../envs/sbs_processing.yml"
    input:
        lambda wildcards: expand(
            PREPROCESS_FP
            / "images"
            / "sbs"
            / get_filename(
                {"well": wildcards.well, "tile": "{tile}", "cycle": wildcards.cycle},
                "image",
                "tiff",
            ),
            tile=SBS_TILES,
        ),
    output:
        SBS_PROCESSING_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tsv"),
    params:
        method="sbs_mean",
        upsample_factor=1,
        display_ranges=config["sbs_processing"]["display_ranges"],
    script:
        "../scripts/sbs_processing/align_cycles.py"
