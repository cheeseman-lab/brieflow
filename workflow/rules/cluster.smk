from lib.shared.compartment_utils import format_rule_output


# bootstrap gene results for the current combo, resolved on the same compartment path
_bootstrap_gene_results = lambda wildcards: format_rule_output(
    BOOTSTRAP_OUTPUTS["combined_gene_results"],
    wildcards,
    SPLIT_BY_COMPARTMENT,
    DEFAULT_COMPARTMENT_COMBO,
)


# Clean aggregate datasets
rule clean_aggregate:
    input:
        ancient(AGGREGATE_OUTPUTS["aggregate"]),
    output:
        CLUSTER_OUTPUTS_MAPPED["clean_aggregate"],
    params:
        cell_class=lambda wildcards: wildcards.cell_class,
        min_cell_cutoffs=config.get("cluster", {}).get("min_cell_cutoffs", {}),
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
        phate_distance_metric=config.get("cluster", {}).get("phate_distance_metric", "cosine"),
        perturbation_name_col=config.get("aggregate", {}).get("perturbation_name_col"),
        control_key=config.get("aggregate", {}).get("control_key"),
        control_scope=config.get("cluster", {}).get("control_scope", "pooled"),
        control_reference_group=config.get("cluster", {}).get("control_reference_group", None),
        group_cols=config.get("aggregate", {}).get("group_cols", []),
        uniprot_data_fp=config.get("cluster", {}).get("uniprot_data_fp"),
        perturbation_auc_threshold=config.get("cluster", {}).get("perturbation_auc_threshold"),
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
        phate_distance_metric=config.get("cluster", {}).get("phate_distance_metric", "cosine"),
        perturbation_name_col=config.get("aggregate", {}).get("perturbation_name_col"),
        control_key=config.get("aggregate", {}).get("control_key"),
        perturbation_auc_threshold=config.get("cluster", {}).get("perturbation_auc_threshold"),
        string_pair_benchmark_fp=config.get("cluster", {}).get("string_pair_benchmark_fp"),
        corum_group_benchmark_fp=config.get("cluster", {}).get("corum_group_benchmark_fp"),
        kegg_group_benchmark_fp=config.get("cluster", {}).get("kegg_group_benchmark_fp"),
    script:
        "../scripts/cluster/benchmark_clusters.py"


# Format cluster results into AnnData h5ad (combines all resolutions)
rule format_cluster_anndata:
    input:
        features_genes=lambda wildcards: format_rule_output(
            AGGREGATE_OUTPUTS["generate_feature_table"][2],
            wildcards,
            SPLIT_BY_COMPARTMENT,
            DEFAULT_COMPARTMENT_COMBO,
        ),
        clustering=lambda wildcards: [
            format_rule_output(
                CLUSTER_OUTPUTS["phate_leiden_clustering"][0],
                wildcards,
                SPLIT_BY_COMPARTMENT,
                DEFAULT_COMPARTMENT_COMBO,
                leiden_resolution=res,
            )
            for res in cluster_wildcard_combos["leiden_resolution"].unique()
        ],
        bootstrap_results=lambda wildcards: (
            ancient(_bootstrap_gene_results(wildcards))
            if Path(_bootstrap_gene_results(wildcards)).exists()
            else []
        ),
    output:
        CLUSTER_OUTPUTS_MAPPED["format_cluster_anndata"],
    params:
        perturbation_name_col=config.get("aggregate", {}).get("perturbation_name_col"),
        channel_names=config.get("phenotype", {}).get("channel_names"),
        leiden_resolutions=list(cluster_wildcard_combos["leiden_resolution"].unique()),
    script:
        "../scripts/cluster/format_cluster_anndata.py"


# Rule for all cluster processing steps
rule all_cluster:
    input:
        CLUSTER_TARGETS_ALL,
