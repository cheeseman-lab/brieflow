control_prefix = "nontargeting"
cutoffs = {"mitotic": 0, "interphase": 3, "all": 3}
channels = ["dapi", "coxiv", "cenpa", "wga"]
channel_pairs = ["all", ("dapi", "coxiv"), ("dapi", "cenpa"), ("dapi", "wga")]

# Analysis parameters
correlation_threshold = 0.99
variance_threshold = 0.001
min_unique_values = 5
leiden_resolution = 5.0

# TODO: use actual input targets


# Generate cluster datasets
rule generate_cluster_datasets:
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
        CLUSTER_PROCESS_OUTPUTS_MAPPED["generate_cluster_datasets"],
    params:
        dataset=lambda wildcards: wildcards.dataset,
        channel_combo=lambda wildcards: wildcards.channel_combo,
        all_channels=config["phenotype_process"]["channel_names"],
        min_cell_cutoffs=config["cluster_process"]["min_cell_cutoffs"],
    script:
        "../scripts/cluster_process/generate_cluster_datasets.py"


# Rule for all cluster processing steps
rule all_cluster_process:
    input:
        CLUSTER_PROCESS_TARGETS_ALL,
