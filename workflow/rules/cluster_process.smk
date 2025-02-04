# TODO: use actual input targets


# Generate cluster datasets
rule generate_dataset:
    conda:
        "../envs/cluster_process.yml"
    input:
        # final gene datasets
        # AGGREGATE_PROCESS_OUTPUTS["process_mitotic_gene_data"],
        # AGGREGATE_PROCESS_OUTPUTS["process_interphase_gene_data"],
        # AGGREGATE_PROCESS_OUTPUTS["process_all_gene_data"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/tsvs/mitotic_gene_data.tsv",
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/tsvs/interphase_gene_data.tsv",
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/tsvs/all_gene_data.tsv",
    output:
        CLUSTER_PROCESS_OUTPUTS_MAPPED["generate_dataset"],
    params:
        dataset=lambda wildcards: wildcards.dataset,
        channel_combo=lambda wildcards: wildcards.channel_combo,
        all_channels=config["phenotype_process"]["channel_names"],
        min_cell_cutoffs=config["cluster_process"]["min_cell_cutoffs"],
    script:
        "../scripts/cluster_process/generate_dataset.py"


# perform phate embedding and leiden clustering
rule phate_leiden_clustering:
    conda:
        "../envs/cluster_process.yml"
    input:
        # cluster dataset
        CLUSTER_PROCESS_OUTPUTS["generate_dataset"],
    output:
        CLUSTER_PROCESS_OUTPUTS_MAPPED["phate_leiden_clustering"],
    params:
        correlation_threshold=config["cluster_process"]["correlation_threshold"],
        variance_threshold=config["cluster_process"]["variance_threshold"],
        min_unique_values=config["cluster_process"]["min_unique_values"],
        control_prefix=config["aggregate_process"]["control_prefix"],
        leiden_resolution=config["cluster_process"]["leiden_resolution"],
        population_feature=config["aggregate_process"]["population_feature"],
        uniprot_data_fp=config["cluster_process"]["uniprot_data_fp"],
    script:
        "../scripts/cluster_process/phate_leiden_clustering.py"


# analyze clusters with uniprot data
rule analyze_clusters:
    conda:
        "../envs/cluster_process.yml"
    input:
        # phate leiden clusters with uniprot data
        CLUSTER_PROCESS_OUTPUTS["phate_leiden_clustering"][2],
        # cleaned gene data
        CLUSTER_PROCESS_OUTPUTS["generate_dataset"],
    output:
        CLUSTER_PROCESS_OUTPUTS_MAPPED["analyze_clusters"],
    params:
        population_feature=config["aggregate_process"]["population_feature"],
        string_data_fp=config["cluster_process"]["string_data_fp"],
        corum_data_fp=config["cluster_process"]["corum_data_fp"],
    script:
        "../scripts/cluster_process/analyze_clusters.py"


# evaluate clustering
rule cluster_eval:
    conda:
        "../envs/cluster_process.yml"
    input:
        # all global metric files from analyze clusters
        lambda wildcards: output_to_input(
            CLUSTER_PROCESS_OUTPUTS["analyze_clusters"][1],
            {"channel_combo": CHANNEL_COMBOS, "dataset": DATASETS},
            wildcards,
        ),
    output:
        CLUSTER_PROCESS_OUTPUTS_MAPPED["cluster_eval"],
    script:
        "../scripts/cluster_process/cluster_eval.py"


# Rule for all cluster processing steps
rule all_cluster_process:
    input:
        CLUSTER_PROCESS_TARGETS_ALL,
