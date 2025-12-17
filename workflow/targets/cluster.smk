import pandas as pd
from itertools import product

from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_FP = ROOT_FP / "cluster"

CLUSTER_OUTPUTS = {
    "clean_aggregate": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / get_filename({}, "aggregate_cleaned", "tsv"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {},
            "phate_leiden_clustering",
            "tsv",
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "cluster_sizes", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "clusters", "png"),
    ],
    # Filtered clustering outputs
    "merge_bootstrap_genes": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / get_filename({}, "merged_bootstrap_genes", "tsv"),
    ],
    "filter_genes": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / get_filename({}, "filtered_genes", "tsv"),
    ],
    "clean_aggregate_filtered": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / get_filename({}, "aggregate_cleaned", "tsv"),
    ],
    "phate_leiden_clustering_filtered": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({}, "phate_leiden_clustering", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({}, "cluster_sizes", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({}, "clusters", "png"),
    ],
    "benchmark_clusters_filtered": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "pie_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "enrichment_pie_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "enrichment_bar_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "enrichment_bar_chart", "png"),
    ],
    # Mozzarellm outputs
    "run_mozzarellm": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / "mozzarellm"
        / get_filename({}, "mozzarellm_clusters", "json"),
    ],
    "run_mozzarellm_filtered": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / "mozzarellm"
        / get_filename({}, "mozzarellm_clusters", "json"),
    ],
    "benchmark_clusters": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "pie_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_pie_chart", "png"
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "enrichment_bar_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_bar_chart", "png"
        ),
    ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "clean_aggregate": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
    # Filtered clustering mappings
    "merge_bootstrap_genes": None,
    "filter_genes": None,
    "clean_aggregate_filtered": None,
    "phate_leiden_clustering_filtered": None,
    "benchmark_clusters_filtered": None,
    # Mozzarellm mappings
    "run_mozzarellm": None,
    "run_mozzarellm_filtered": None,
}


# TODO: Use all combos
# cluster_wildcard_combos = cluster_wildcard_combos[
#     (cluster_wildcard_combos["cell_class"].isin(["Interphase"]))
#     & (cluster_wildcard_combos["channel_combo"].isin(["DAPI_COXIV_CENPA_WGA"]))
#     & (cluster_wildcard_combos["leiden_resolution"].isin([13]))
# ]

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

# Base clustering outputs (conditionally included)
BASE_CLUSTER_OUTPUTS = {
    "clean_aggregate": CLUSTER_OUTPUTS["clean_aggregate"],
    "phate_leiden_clustering": CLUSTER_OUTPUTS["phate_leiden_clustering"],
    "benchmark_clusters": CLUSTER_OUTPUTS["benchmark_clusters"],
}
BASE_CLUSTER_MAPPINGS = {
    "clean_aggregate": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
}

CLUSTER_TARGETS_ALL = []

# Add standard (unfiltered) clustering targets if enabled
standard_clustering_config = config.get("cluster", {}).get("standard_clustering", {})
if standard_clustering_config.get("enabled", True):  # Default True for backwards compatibility
    # Load standard cluster combos from file
    standard_combo_fp = standard_clustering_config.get("cluster_combo_fp")
    if standard_combo_fp:
        standard_combos = pd.read_csv(standard_combo_fp, sep="\t")
    else:
        # Fallback to legacy cluster_combo_fp for backwards compatibility
        standard_combos = cluster_wildcard_combos

    CLUSTER_TARGETS_ALL += outputs_to_targets(
        BASE_CLUSTER_OUTPUTS, standard_combos, BASE_CLUSTER_MAPPINGS
    )

# Add filtered clustering targets if enabled
filtered_clustering_config = config.get("cluster", {}).get("filtered_clustering", {})
if filtered_clustering_config.get("enabled", False):
    FILTERED_CLUSTER_OUTPUTS = {
        "merge_bootstrap_genes": CLUSTER_OUTPUTS["merge_bootstrap_genes"],
        "filter_genes": CLUSTER_OUTPUTS["filter_genes"],
        "clean_aggregate_filtered": CLUSTER_OUTPUTS["clean_aggregate_filtered"],
        "phate_leiden_clustering_filtered": CLUSTER_OUTPUTS["phate_leiden_clustering_filtered"],
        "benchmark_clusters_filtered": CLUSTER_OUTPUTS["benchmark_clusters_filtered"],
    }
    FILTERED_CLUSTER_MAPPINGS = {
        "merge_bootstrap_genes": None,
        "filter_genes": None,
        "clean_aggregate_filtered": None,
        "phate_leiden_clustering_filtered": None,
        "benchmark_clusters_filtered": None,
    }

    # Load filtered cluster combos from file
    filtered_combo_fp = filtered_clustering_config.get("cluster_filtered_combo_fp")
    if filtered_combo_fp:
        filtered_combos = pd.read_csv(filtered_combo_fp, sep="\t")
    else:
        # Fallback to standard combos if no filtered-specific file
        filtered_combos = cluster_wildcard_combos

    CLUSTER_TARGETS_ALL += outputs_to_targets(
        FILTERED_CLUSTER_OUTPUTS, filtered_combos, FILTERED_CLUSTER_MAPPINGS
    )

# Add mozzarellm targets based on specific combinations from config
mozzarellm_config = config.get("cluster", {}).get("mozzarellm", {})
if mozzarellm_config.get("enabled", False):
    mozzarellm_combinations = mozzarellm_config.get("mozzarellm_combinations", [])

    for combo in mozzarellm_combinations:
        cell_class = combo.get("cell_class")
        channel_combo = combo.get("channel_combo")
        leiden_resolution = combo.get("leiden_resolution")
        use_filtered = combo.get("use_filtered", False)

        if use_filtered and filtered_clustering_config.get("enabled", False):
            # Use filtered clustering output path
            target_path = str(CLUSTER_OUTPUTS["run_mozzarellm_filtered"][0]).format(
                channel_combo=channel_combo,
                cell_class=cell_class,
                leiden_resolution=leiden_resolution,
            )
        else:
            # Use unfiltered clustering output path
            target_path = str(CLUSTER_OUTPUTS["run_mozzarellm"][0]).format(
                channel_combo=channel_combo,
                cell_class=cell_class,
                leiden_resolution=leiden_resolution,
            )

        CLUSTER_TARGETS_ALL.append(target_path)
