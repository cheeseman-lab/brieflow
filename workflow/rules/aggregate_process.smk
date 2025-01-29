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
    input:
        # standardized data
        # AGGREGATE_PROCESS_OUTPUTS["standardize_features"],
        "/lab/barcheese01/rkern/brieflow/example_analysis/analysis_root/aggregate_process/hdfs/transformed_data.hdf5",
    output:
        "aggregate_4/hdf/mitotic_data.hdf",
        "aggregate_4/hdf/interphase_data.hdf",
    resources:
        mem_mb=500000,
    priority: 1
    run:
        df = pd.read_hdf(input[0])
        mitotic_df, interphase_df = split_mitotic_simple(df, THRESHOLDS)

        mitotic_df.to_hdf(output[0], key="data", mode="w")
        interphase_df.to_hdf(output[1], key="data", mode="w")


# Rule for all aggregate processing steps
rule all_aggregate_process:
    input:
        AGGREGATE_PROCESS_TARGETS_ALL,
