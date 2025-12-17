import os

# Wildcard constraints to prevent ambiguous matching
wildcard_constraints:
    cell_class="[^/]+",
    channel_combo="[^/]+",
    leiden_resolution="[^/]+",
    cluster_id="[^/]+",


# Clean aggregate datasets
rule clean_aggregate:
    input:
        ancient(AGGREGATE_OUTPUTS["aggregate"]),
    output:
        CLUSTER_OUTPUTS_MAPPED["clean_aggregate"],
    params:
        cell_class=lambda wildcards: wildcards.cell_class,
        min_cell_cutoffs=config["cluster"]["min_cell_cutoffs"],
    script:
        "../scripts/cluster/clean_aggregate.py"


# perform phate embedding and leiden clustering
rule phate_leiden_clustering:
    input:
        CLUSTER_OUTPUTS["clean_aggregate"],
    output:
        CLUSTER_OUTPUTS_MAPPED["phate_leiden_clustering"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        phate_distance_metric=config["cluster"]["phate_distance_metric"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_key=config["aggregate"]["control_key"],
        uniprot_data_fp=config["cluster"]["uniprot_data_fp"],
        perturbation_auc_threshold=config["cluster"]["perturbation_auc_threshold"],
    script:
        "../scripts/cluster/phate_leiden_clustering.py"


# benchmark
rule benchmark_clusters:
    input:
        CLUSTER_OUTPUTS["clean_aggregate"],
        CLUSTER_OUTPUTS["phate_leiden_clustering"][0],
    output:
        CLUSTER_OUTPUTS_MAPPED["benchmark_clusters"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        phate_distance_metric=config["cluster"]["phate_distance_metric"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_key=config["aggregate"]["control_key"],
        string_pair_benchmark_fp=config["cluster"]["string_pair_benchmark_fp"],
        corum_group_benchmark_fp=config["cluster"]["corum_group_benchmark_fp"],
        kegg_group_benchmark_fp=config["cluster"]["kegg_group_benchmark_fp"],
    script:
        "../scripts/cluster/benchmark_clusters.py"


# =============================================================================
# FILTERED CLUSTERING RULES
# These rules implement a parallel clustering arm that filters genes based on
# bootstrap statistics (Z-score and/or FDR) before clustering
# =============================================================================


# Merge bootstrap results with gene-level features
rule merge_bootstrap_genes:
    input:
        bootstrap=ancient(BOOTSTRAP_OUTPUTS["combined_gene_results"]),
        genes=ancient(AGGREGATE_OUTPUTS["generate_feature_table"][2]),
    output:
        CLUSTER_OUTPUTS_MAPPED["merge_bootstrap_genes"],
    params:
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
    script:
        "../scripts/cluster/merge_bootstrap_genes.py"


# Filter genes based on bootstrap statistics and subset aggregated data
rule filter_genes:
    input:
        merged_bootstrap=CLUSTER_OUTPUTS["merge_bootstrap_genes"],
        aggregated=ancient(AGGREGATE_OUTPUTS["aggregate"]),
    output:
        CLUSTER_OUTPUTS_MAPPED["filter_genes"],
    params:
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_patterns=config["cluster"]["filtered_clustering"]["control_patterns"],
        zscore_threshold=config["cluster"]["filtered_clustering"]["zscore_threshold"],
        zscore_direction=config["cluster"]["filtered_clustering"]["zscore_direction"],
        fdr_threshold=config["cluster"]["filtered_clustering"]["fdr_threshold"],
        filter_mode=config["cluster"]["filtered_clustering"]["filter_mode"],
    script:
        "../scripts/cluster/filter_genes.py"


# Clean aggregate for filtered genes
rule clean_aggregate_filtered:
    input:
        CLUSTER_OUTPUTS["filter_genes"],
    output:
        CLUSTER_OUTPUTS_MAPPED["clean_aggregate_filtered"],
    params:
        cell_class=lambda wildcards: wildcards.cell_class,
        min_cell_cutoffs=config["cluster"]["min_cell_cutoffs"],
    script:
        "../scripts/cluster/clean_aggregate.py"


# PHATE + Leiden clustering on filtered genes
rule phate_leiden_clustering_filtered:
    input:
        CLUSTER_OUTPUTS["clean_aggregate_filtered"],
    output:
        CLUSTER_OUTPUTS_MAPPED["phate_leiden_clustering_filtered"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        phate_distance_metric=config["cluster"]["phate_distance_metric"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_key=config["aggregate"]["control_key"],
        uniprot_data_fp=config["cluster"]["uniprot_data_fp"],
        perturbation_auc_threshold=config["cluster"]["perturbation_auc_threshold"],
    script:
        "../scripts/cluster/phate_leiden_clustering.py"


# Benchmark filtered clusters
rule benchmark_clusters_filtered:
    input:
        CLUSTER_OUTPUTS["clean_aggregate_filtered"],
        CLUSTER_OUTPUTS["phate_leiden_clustering_filtered"][0],
    output:
        CLUSTER_OUTPUTS_MAPPED["benchmark_clusters_filtered"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        phate_distance_metric=config["cluster"]["phate_distance_metric"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_key=config["aggregate"]["control_key"],
        string_pair_benchmark_fp=config["cluster"]["string_pair_benchmark_fp"],
        corum_group_benchmark_fp=config["cluster"]["corum_group_benchmark_fp"],
        kegg_group_benchmark_fp=config["cluster"]["kegg_group_benchmark_fp"],
    script:
        "../scripts/cluster/benchmark_clusters.py"


# =============================================================================
# MOZZARELLM RULES
# LLM-based cluster analysis using the mozzarellm package
# Uses checkpoint pattern to parallelize across clusters within each clustering
# =============================================================================


# Helper function to get cluster result files from checkpoint output
def get_mozzarellm_cluster_results(wildcards):
    """Get all cluster result files after checkpoint completes."""
    checkpoint_output = checkpoints.prepare_mozzarellm_jobs.get(**wildcards).output[0]
    cluster_ids = glob_wildcards(
        os.path.join(checkpoint_output, "cluster_{cluster_id}.json")
    ).cluster_id
    return expand(
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / "mozzarellm"
        / "clusters"
        / "cluster_{cluster_id}.json",
        channel_combo=wildcards.channel_combo,
        cell_class=wildcards.cell_class,
        leiden_resolution=wildcards.leiden_resolution,
        cluster_id=cluster_ids,
    )


def get_mozzarellm_filtered_cluster_results(wildcards):
    """Get all cluster result files after checkpoint completes (filtered)."""
    checkpoint_output = checkpoints.prepare_mozzarellm_jobs_filtered.get(
        **wildcards
    ).output[0]
    cluster_ids = glob_wildcards(
        os.path.join(checkpoint_output, "cluster_{cluster_id}.json")
    ).cluster_id
    return expand(
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "filtered"
        / "{leiden_resolution}"
        / "mozzarellm"
        / "clusters"
        / "cluster_{cluster_id}.json",
        channel_combo=wildcards.channel_combo,
        cell_class=wildcards.cell_class,
        leiden_resolution=wildcards.leiden_resolution,
        cluster_id=cluster_ids,
    )


# Checkpoint: Prepare job files for each cluster (unfiltered)
checkpoint prepare_mozzarellm_jobs:
    input:
        cluster_file=CLUSTER_OUTPUTS["phate_leiden_clustering"][0],
    output:
        directory(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "jobs"
        ),
    script:
        "../scripts/cluster/prepare_mozzarellm_jobs.py"


# Checkpoint: Prepare job files for each cluster (filtered)
checkpoint prepare_mozzarellm_jobs_filtered:
    input:
        cluster_file=CLUSTER_OUTPUTS["phate_leiden_clustering_filtered"][0],
    output:
        directory(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "filtered"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "jobs"
        ),
    script:
        "../scripts/cluster/prepare_mozzarellm_jobs.py"


# Run mozzarellm on a single cluster (unfiltered)
rule run_mozzarellm_cluster:
    input:
        job_file=(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "jobs"
            / "cluster_{cluster_id}.json"
        ),
    output:
        json_output=(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "clusters"
            / "cluster_{cluster_id}.json"
        ),
    params:
        model_name=config["cluster"]["mozzarellm"]["model_name"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        screen_context=config["cluster"]["mozzarellm"].get("screen_context"),
        cluster_analysis_prompt=config["cluster"]["mozzarellm"].get(
            "cluster_analysis_prompt"
        ),
    script:
        "../scripts/cluster/run_mozzarellm_cluster.py"


# Run mozzarellm on a single cluster (filtered)
rule run_mozzarellm_cluster_filtered:
    input:
        job_file=(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "filtered"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "jobs"
            / "cluster_{cluster_id}.json"
        ),
    output:
        json_output=(
            CLUSTER_FP
            / "{channel_combo}"
            / "{cell_class}"
            / "filtered"
            / "{leiden_resolution}"
            / "mozzarellm"
            / "clusters"
            / "cluster_{cluster_id}.json"
        ),
    params:
        model_name=config["cluster"]["mozzarellm"]["model_name"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        screen_context=config["cluster"]["mozzarellm"].get("screen_context"),
        cluster_analysis_prompt=config["cluster"]["mozzarellm"].get(
            "cluster_analysis_prompt"
        ),
    script:
        "../scripts/cluster/run_mozzarellm_cluster.py"


# Combine mozzarellm results from all clusters (unfiltered)
rule combine_mozzarellm:
    input:
        cluster_results=get_mozzarellm_cluster_results,
    output:
        CLUSTER_OUTPUTS_MAPPED["run_mozzarellm"],
    params:
        model_name=config["cluster"]["mozzarellm"]["model_name"],
    script:
        "../scripts/cluster/combine_mozzarellm.py"


# Combine mozzarellm results from all clusters (filtered)
rule combine_mozzarellm_filtered:
    input:
        cluster_results=get_mozzarellm_filtered_cluster_results,
    output:
        CLUSTER_OUTPUTS_MAPPED["run_mozzarellm_filtered"],
    params:
        model_name=config["cluster"]["mozzarellm"]["model_name"],
    script:
        "../scripts/cluster/combine_mozzarellm.py"


# Rule for all cluster processing steps
rule all_cluster:
    input:
        CLUSTER_TARGETS_ALL,
