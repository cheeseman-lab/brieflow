from lib.mozzarellm.annotate_clusters import mozzarellm_row_value
from lib.shared.compartment_utils import format_rule_output


# the clustering h5ad for the current combo, resolved on the same compartment path
_cluster_anndata = lambda wildcards: format_rule_output(
    CLUSTER_OUTPUTS["format_cluster_anndata"][0],
    wildcards,
    SPLIT_BY_COMPARTMENT,
    DEFAULT_COMPARTMENT_COMBO,
)


# the screen context the combo table pairs with this clustering, written before the run
_screen_context = lambda wildcards: mozzarellm_row_value(
    mozzarellm_wildcard_combos,
    "screen_context_fp",
    wildcards,
    SPLIT_BY_COMPARTMENT,
    DEFAULT_COMPARTMENT_COMBO,
)


# annotate one clustering's clusters with mozzarellm
rule annotate_clusters:
    input:
        cluster_anndata=_cluster_anndata,
        screen_context=_screen_context,
    output:
        MOZZARELLM_OUTPUTS_MAPPED["annotate_clusters"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        run_name=MOZZARELLM_RUN_NAME,
        screen_name=MOZZARELLM_SCREEN_NAME,
        model=config.get("mozzarellm", {}).get("model", "claude-sonnet-5"),
        mode=config.get("mozzarellm", {}).get("mode", "cot"),
        mcp=config.get("mozzarellm", {}).get("mcp", True),
        source=config.get("mozzarellm", {}).get("source", "affinage"),
        include_features=config.get("mozzarellm", {}).get("include_features", "auto"),
        include_strength=config.get("mozzarellm", {}).get("include_strength", "auto"),
        n_features=config.get("mozzarellm", {}).get("n_features", 5),
        fdr_threshold=config.get("mozzarellm", {}).get("fdr_threshold", None),
        max_tokens=config.get("mozzarellm", {}).get("max_tokens", 64000),
        max_workers=MOZZARELLM_MAX_WORKERS,
        max_failed_clusters=MOZZARELLM_MAX_FAILED_CLUSTERS,
    threads: MOZZARELLM_MAX_WORKERS
    resources:
        mem_mb=4000,
        runtime=MOZZARELLM_RUNTIME,
    retries: 2
    script:
        "../scripts/mozzarellm/annotate_clusters.py"


# merge the run's per-gene and per-cluster calls into the clustering h5ad
rule annotate_cluster_anndata:
    input:
        cluster_anndata=_cluster_anndata,
        clusters_json=MOZZARELLM_OUTPUTS["annotate_clusters"][0],
        genes_csv=MOZZARELLM_OUTPUTS["annotate_clusters"][1],
    output:
        MOZZARELLM_OUTPUTS_MAPPED["annotate_cluster_anndata"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        run_name=MOZZARELLM_RUN_NAME,
        screen_name=MOZZARELLM_SCREEN_NAME,
    script:
        "../scripts/mozzarellm/annotate_cluster_anndata.py"


# Rule for all mozzarellm annotation steps
rule all_mozzarellm:
    input:
        MOZZARELLM_TARGETS_ALL,
