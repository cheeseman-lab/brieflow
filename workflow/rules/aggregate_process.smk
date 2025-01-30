from lib.shared.target_utils import output_to_input

# TODO: update input paths to reflect dynamic target paths
# TODO: remove threads references here


# Clean and transform merged data
rule clean_and_transform:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # final merge data
        # MERGE_PROCESS_OUTPUTS["final_merge"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/merge_process/hdfs/merge_final.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["clean_and_transform"],
    params:
        population_feature=config["aggregate_process"]["population_feature"],
        transformations_fp=config["aggregate_process"]["transformations_fp"],
        channels=config["phenotype_process"]["channel_names"],
    script:
        "../scripts/aggregate_process/clean_and_transform.py"


# Standardize features
rule standardize_features:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # final merge data
        # AGGREGATE_PROCESS_OUTPUTS["clean_and_transform"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/transformed_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["standardize_features"],
    params:
        feature_start=config["aggregate_process"]["feature_start"],
        population_feature=config["aggregate_process"]["population_feature"],
        control_prefix=config["aggregate_process"]["control_prefix"],
        group_columns=config["aggregate_process"]["group_columns"],
        index_columns=config["aggregate_process"]["index_columns"],
        cat_columns=config["aggregate_process"]["cat_columns"],
    script:
        "../scripts/aggregate_process/standardize_features.py"


# Split mitotic and interphase data
rule split_phases:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # standardized data
        # AGGREGATE_PROCESS_OUTPUTS["standardize_features"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/standardized_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["split_phases"],
    params:
        threshold_conditions=config["aggregate_process"]["threshold_conditions"],
    script:
        "../scripts/aggregate_process/split_phases.py"


# Process mitotic gene data
rule process_mitotic_gene_data:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # mitotic data
        # AGGREGATE_PROCESS_OUTPUTS["split_phases"][0]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/mitotic_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["process_mitotic_gene_data"],
    params:
        standardize_data=True,
        feature_start=config["aggregate_process"]["feature_start"],
        population_feature=config["aggregate_process"]["population_feature"],
        control_prefix=config["aggregate_process"]["control_prefix"],
        group_columns=config["aggregate_process"]["group_columns"],
        index_columns=config["aggregate_process"]["index_columns"],
        cat_columns=config["aggregate_process"]["cat_columns"],
    script:
        "../scripts/aggregate_process/process_gene_data.py"


# Process interphase gene data
rule process_interphase_gene_data:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # interphase data
        # AGGREGATE_PROCESS_OUTPUTS["split_phases"][1]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/interphase_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["process_interphase_gene_data"],
    params:
        standardize_data=True,
        feature_start=config["aggregate_process"]["feature_start"],
        population_feature=config["aggregate_process"]["population_feature"],
        control_prefix=config["aggregate_process"]["control_prefix"],
        group_columns=config["aggregate_process"]["group_columns"],
        index_columns=config["aggregate_process"]["index_columns"],
        cat_columns=config["aggregate_process"]["cat_columns"],
    script:
        "../scripts/aggregate_process/process_gene_data.py"


# Process all gene data
rule process_all_gene_data:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # all standardized data
        # AGGREGATE_PROCESS_OUTPUTS["standardize_features"]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/standardized_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["process_all_gene_data"],
    params:
        standardize_data=False,
        feature_start=config["aggregate_process"]["feature_start"],
        population_feature=config["aggregate_process"]["population_feature"],
        control_prefix=config["aggregate_process"]["control_prefix"],
        group_columns=config["aggregate_process"]["group_columns"],
        index_columns=config["aggregate_process"]["index_columns"],
        cat_columns=config["aggregate_process"]["cat_columns"],
    script:
        "../scripts/aggregate_process/process_gene_data.py"


# Prepare mitotic montage data
rule prepare_mitotic_montage_data:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # all standardized data
        # AGGREGATE_PROCESS_OUTPUTS["split_phases"][0]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/mitotic_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["prepare_mitotic_montage_data"],
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate_process/prepare_montage_data.py"


# Prepare interphase montage data
rule prepare_interphase_montage_data:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # all standardized data
        # AGGREGATE_PROCESS_OUTPUTS["split_phases"][1]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/interphase_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["prepare_interphase_montage_data"],
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate_process/prepare_montage_data.py"


# TODO: Optimize montages generation to operate faster! We should try to:
# 1. Restrict montage generation attempts to those that we have data for
# 2. Save each channel montage for a gene/sgrna pair during one rule call
# 3. Possibly parallelize across a rule so that we only need to load cell data once


# Create mitotic montages data
rule generate_mitotic_montage:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # mitotic montage data
        # AGGREGATE_PROCESS_OUTPUTS["prepare_mitotic_montage_data"]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/mitotic_montage_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["generate_mitotic_montage"],
    params:
        channels=config["phenotype_process"]["channel_names"],
    script:
        "../scripts/aggregate_process/generate_montage.py"


# Create interphase montages data
rule generate_interphase_montage:
    conda:
        "../envs/aggregate_process.yml"
    input:
        # mitotic montage data
        # AGGREGATE_PROCESS_OUTPUTS["prepare_interphase_montage_data"]
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/interphase_montage_data.hdf5",
    output:
        AGGREGATE_PROCESS_OUTPUTS_MAPPED["generate_interphase_montage"],
    params:
        channels=config["phenotype_process"]["channel_names"],
    script:
        "../scripts/aggregate_process/generate_montage.py"


# Rule for all aggregate processing steps
rule all_aggregate_process:
    input:
        AGGREGATE_PROCESS_TARGETS_ALL,
