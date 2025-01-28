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
        population_feature=config["aggregate_process"]["population_feature"],
        channels=config["phenotype_process"]["channel_names"],
    script:
        "../scripts/aggregate_process/standardize_features.py"
