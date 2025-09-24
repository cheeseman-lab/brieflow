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
# rule benchmark_clusters:
#     input:
#         CLUSTER_OUTPUTS["clean_aggregate"],
#         CLUSTER_OUTPUTS["phate_leiden_clustering"][0],
#     output:
#         CLUSTER_OUTPUTS_MAPPED["benchmark_clusters"],
#     params:
#         leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
#         phate_distance_metric=config["cluster"]["phate_distance_metric"],
#         perturbation_name_col=config["aggregate"]["perturbation_name_col"],
#         control_key=config["aggregate"]["control_key"],
#         string_pair_benchmark_fp=config["cluster"]["string_pair_benchmark_fp"],
#         corum_group_benchmark_fp=config["cluster"]["corum_group_benchmark_fp"],
#         kegg_group_benchmark_fp=config["cluster"]["kegg_group_benchmark_fp"],
#     script:
#         "../scripts/cluster/benchmark_clusters.py"


# Rule for all cluster processing steps
rule all_cluster:
    input:
        CLUSTER_TARGETS_ALL,
