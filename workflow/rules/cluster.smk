# Clean aggregate datasets
rule clean_aggregate:
    input:
        # aggregated data
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
        # cluster dataset
        CLUSTER_OUTPUTS["clean_aggregate"],
    output:
        CLUSTER_OUTPUTS_MAPPED["phate_leiden_clustering"],
    params:
        leiden_resolution=lambda wildcards: wildcards.leiden_resolution,
        phate_distance_metric=config["cluster"]["phate_distance_metric"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        control_key=config["aggregate"]["control_key"],
    script:
        "../scripts/cluster/phate_leiden_clustering.py"


# # analyze clusters with uniprot data
# rule benchmark_clusters:
#     input:
#         # phate leiden clusters with uniprot data
#         CLUSTER_OUTPUTS["phate_leiden_clustering"][2],
#         # cleaned gene data
#         CLUSTER_OUTPUTS["generate_dataset"],
#     output:
#         CLUSTER_OUTPUTS_MAPPED["benchmark_clusters"],
#     params:
#         population_feature=config["aggregate"]["population_feature"],
#         string_data_fp=config["cluster"]["string_data_fp"],
#         corum_data_fp=config["cluster"]["corum_data_fp"],
#     script:
#         "../scripts/cluster/benchmark_clusters.py"


# # evaluate clustering
# rule cluster_eval:
#     input:
#         # all global metric files from analyze clusters
#         lambda wildcards: output_to_input(
#             CLUSTER_OUTPUTS["benchmark_clusters"][1],
#             wildcards=wildcards,
#             expansion_values=["channel_combo", "dataset"],
#             metadata_combos=cluster_wildcard_combos,
#         ),
#     output:
#         CLUSTER_OUTPUTS_MAPPED["cluster_eval"],
#     script:
#         "../scripts/cluster/cluster_eval.py"


# Rule for all cluster processing steps
rule all_cluster:
    input:
        CLUSTER_TARGETS_ALL,
