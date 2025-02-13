# Generate cluster datasets
rule generate_dataset:
    conda:
        "../envs/cluster.yml"
    input:
        # final gene datasets
        AGGREGATE_OUTPUTS["process_mitotic_gene_data"],
        AGGREGATE_OUTPUTS["process_interphase_gene_data"],
        AGGREGATE_OUTPUTS["process_all_gene_data"],
    output:
        CLUSTER_OUTPUTS_MAPPED["generate_dataset"],
    params:
        dataset=lambda wildcards: wildcards.dataset,
        channel_combo=lambda wildcards: wildcards.channel_combo,
        all_channels=config["phenotype"]["channel_names"],
        min_cell_cutoffs=config["cluster"]["min_cell_cutoffs"],
    script:
        "../scripts/cluster/generate_dataset.py"


# perform phate embedding and leiden clustering
rule phate_leiden_clustering:
    conda:
        "../envs/cluster.yml"
    input:
        # cluster dataset
        CLUSTER_OUTPUTS["generate_dataset"],
    output:
        CLUSTER_OUTPUTS_MAPPED["phate_leiden_clustering"],
    params:
        correlation_threshold=config["cluster"]["correlation_threshold"],
        variance_threshold=config["cluster"]["variance_threshold"],
        min_unique_values=config["cluster"]["min_unique_values"],
        control_prefix=config["aggregate"]["control_prefix"],
        cum_var_threshold=config["cluster"]["cum_var_threshold"],
        leiden_resolution=config["cluster"]["leiden_resolution"],
        population_feature=config["aggregate"]["population_feature"],
        uniprot_data_fp=config["cluster"]["uniprot_data_fp"],
    script:
        "../scripts/cluster/phate_leiden_clustering.py"


# analyze clusters with uniprot data
rule benchmark_clusters:
    conda:
        "../envs/cluster.yml"
    input:
        # phate leiden clusters with uniprot data
        CLUSTER_OUTPUTS["phate_leiden_clustering"][2],
        # cleaned gene data
        CLUSTER_OUTPUTS["generate_dataset"],
    output:
        CLUSTER_OUTPUTS_MAPPED["benchmark_clusters"],
    params:
        population_feature=config["aggregate"]["population_feature"],
        string_data_fp=config["cluster"]["string_data_fp"],
        corum_data_fp=config["cluster"]["corum_data_fp"],
    script:
        "../scripts/cluster/benchmark_clusters.py"


# evaluate clustering
rule cluster_eval:
    conda:
        "../envs/cluster.yml"
    input:
        # all global metric files from analyze clusters
        lambda wildcards: output_to_input(
            CLUSTER_OUTPUTS["benchmark_clusters"][1],
            {"channel_combo": CHANNEL_COMBOS, "dataset": DATASETS},
            wildcards,
        ),
    output:
        CLUSTER_OUTPUTS_MAPPED["cluster_eval"],
    script:
        "../scripts/cluster/cluster_eval.py"


# Rule for all cluster processing steps
rule all_cluster:
    input:
        CLUSTER_TARGETS_ALL,
