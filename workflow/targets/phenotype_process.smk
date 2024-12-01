from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


PHENOTYPE_PROCESS_FP = ROOT_FP / "phenotype_process"

PHENOTYPE_PROCESS_OUTPUTS = {
    "apply_ic_field": [
        PHENOTYPE_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    ],
}

PHENOTYPE_PROCESS_OUTPUT_MAPPINGS = {
    "apply_ic_field": None,
}

PHENOTYPE_PROCESS_WILDCARDS = {
    "well": PHENOTYPE_WELLS,
    "tile": PHENOTYPE_TILES,
}

PHENOTYPE_PROCESS_OUTPUTS_MAPPED = map_outputs(
    PHENOTYPE_PROCESS_OUTPUTS, PHENOTYPE_PROCESS_OUTPUT_MAPPINGS
)

PHENOTYPE_PROCESS_TARGETS = outputs_to_targets(
    PHENOTYPE_PROCESS_OUTPUTS, PHENOTYPE_PROCESS_WILDCARDS
)

PHENOTYPE_PROCESS_TARGETS_ALL = sum(PHENOTYPE_PROCESS_TARGETS.values(), [])
