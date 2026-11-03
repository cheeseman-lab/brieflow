from lib.shared.compartment_utils import add_compartment_path
from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MOZZARELLM_RUN_NAME = config.get("mozzarellm", {}).get("run_name", "run1")
MOZZARELLM_SCREEN_NAME = config.get("mozzarellm", {}).get("screen_name", "screen")
MOZZARELLM_MAX_WORKERS = config.get("mozzarellm", {}).get("max_workers", 8)
# unrecoverable clusters a panel may lose before the job fails and discards its outputs
MOZZARELLM_MAX_FAILED_CLUSTERS = config.get("mozzarellm", {}).get(
    "max_failed_clusters", 2
)
# minutes; a panel answers hundreds of clusters, so the profile default is far too short
MOZZARELLM_RUNTIME = config.get("mozzarellm", {}).get("runtime", 720)

# the columns that name a clustering; the table's other columns are per-row parameters
MOZZARELLM_WILDCARDS = [
    "cell_class",
    "channel_combo",
    "compartment_combo",
    "leiden_resolution",
]

MOZZARELLM_OUTPUT_BASE = (
    add_compartment_path(
        CLUSTER_FP / "{channel_combo}",
        "{compartment_combo}",
        SPLIT_BY_COMPARTMENT,
    )
    / "{cell_class}"
    / "{leiden_resolution}"
)
MOZZARELLM_RUN_BASE = MOZZARELLM_OUTPUT_BASE / "mozzarellm" / MOZZARELLM_RUN_NAME

MOZZARELLM_OUTPUTS = {
    "annotate_clusters": [
        MOZZARELLM_RUN_BASE / f"{MOZZARELLM_SCREEN_NAME}_clusters.json",
        MOZZARELLM_RUN_BASE / f"{MOZZARELLM_SCREEN_NAME}_genes.csv",
    ],
    "annotate_cluster_anndata": [
        MOZZARELLM_OUTPUT_BASE / get_filename({}, "cluster_annotated", "h5ad"),
    ],
}

MOZZARELLM_OUTPUT_MAPPINGS = {
    "annotate_clusters": None,
    "annotate_cluster_anndata": None,
}

MOZZARELLM_OUTPUTS_MAPPED = map_outputs(MOZZARELLM_OUTPUTS, MOZZARELLM_OUTPUT_MAPPINGS)

MOZZARELLM_TARGETS_ALL = outputs_to_targets(
    MOZZARELLM_OUTPUTS,
    mozzarellm_wildcard_combos[MOZZARELLM_WILDCARDS],
    MOZZARELLM_OUTPUT_MAPPINGS,
)
