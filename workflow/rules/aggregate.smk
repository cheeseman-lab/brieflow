from lib.shared.target_utils import output_to_input


# Clean, transform, and standardize merged data
rule clean_transform_standardize:
    conda:
        "../envs/aggregate.yml"
    input:
        # final merge data
        MERGE_OUTPUTS["final_merge"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["clean_transform_standardize"],
    params:
        population_feature=config["aggregate"]["population_feature"],
        filter_single_gene=config["aggregate"]["filter_single_gene"],
        transformations_fp=config["aggregate"]["transformations_fp"],
        channels=config["phenotype"]["channel_names"],
        feature_start=config["aggregate"]["feature_start"],
        control_prefix=config["aggregate"]["control_prefix"],
        group_columns=config["aggregate"]["group_columns"],
        index_columns=config["aggregate"]["index_columns"],
        cat_columns=config["aggregate"]["cat_columns"],
    script:
        "../scripts/aggregate/clean_transform_standardize.py"


# Split mitotic and interphase data
rule split_phases:
    conda:
        "../envs/aggregate.yml"
    input:
        # standardized data
        AGGREGATE_OUTPUTS["clean_transform_standardize"][2],
    output:
        AGGREGATE_OUTPUTS_MAPPED["split_phases"],
    params:
        threshold_conditions=config["aggregate"]["threshold_conditions"],
    script:
        "../scripts/aggregate/split_phases.py"


# Process mitotic gene data
rule process_mitotic_gene_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # mitotic data
        AGGREGATE_OUTPUTS["split_phases"][0],
    output:
        AGGREGATE_OUTPUTS_MAPPED["process_mitotic_gene_data"],
    params:
        standardize_data=True,
        feature_start=config["aggregate"]["feature_start"],
        population_feature=config["aggregate"]["population_feature"],
        control_prefix=config["aggregate"]["control_prefix"],
        group_columns=config["aggregate"]["group_columns"],
        index_columns=config["aggregate"]["index_columns"],
        cat_columns=config["aggregate"]["cat_columns"],
    script:
        "../scripts/aggregate/process_gene_data.py"


# Process interphase gene data
rule process_interphase_gene_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # interphase data
        AGGREGATE_OUTPUTS["split_phases"][1],
    output:
        AGGREGATE_OUTPUTS_MAPPED["process_interphase_gene_data"],
    params:
        standardize_data=True,
        feature_start=config["aggregate"]["feature_start"],
        population_feature=config["aggregate"]["population_feature"],
        control_prefix=config["aggregate"]["control_prefix"],
        group_columns=config["aggregate"]["group_columns"],
        index_columns=config["aggregate"]["index_columns"],
        cat_columns=config["aggregate"]["cat_columns"],
    script:
        "../scripts/aggregate/process_gene_data.py"


# Process all gene data
rule process_all_gene_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # all standardized data
        AGGREGATE_OUTPUTS["clean_transform_standardize"][2],
    output:
        AGGREGATE_OUTPUTS_MAPPED["process_all_gene_data"],
    params:
        standardize_data=False,
        feature_start=config["aggregate"]["feature_start"],
        population_feature=config["aggregate"]["population_feature"],
        control_prefix=config["aggregate"]["control_prefix"],
        group_columns=config["aggregate"]["group_columns"],
        index_columns=config["aggregate"]["index_columns"],
        cat_columns=config["aggregate"]["cat_columns"],
    script:
        "../scripts/aggregate/process_gene_data.py"


# Prepare mitotic montage data
rule prepare_mitotic_montage_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # mitotic standardized data
        AGGREGATE_OUTPUTS["split_phases"][0],
    output:
        AGGREGATE_OUTPUTS_MAPPED["prepare_mitotic_montage_data"],
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate/prepare_montage_data.py"


# Prepare interphase montage data
rule prepare_interphase_montage_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # interphase standardized data
        AGGREGATE_OUTPUTS["split_phases"][1],
    output:
        AGGREGATE_OUTPUTS_MAPPED["prepare_interphase_montage_data"],
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate/prepare_montage_data.py"


# TODO: Optimize montage generation to operate faster! We should try to:
# 1. Restrict montage generation attempts to those that we have data for
# 2. Save each channel montage for a gene/sgrna pair during one rule call
# 3. Possibly parallelize across a rule so that we only need to load cell data once


# Create mitotic montages data
rule generate_mitotic_montage:
    conda:
        "../envs/aggregate.yml"
    input:
        # mitotic montage data
        AGGREGATE_OUTPUTS["prepare_mitotic_montage_data"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["generate_mitotic_montage"],
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/generate_montage.py"


# Create interphase montages data
rule generate_interphase_montage:
    conda:
        "../envs/aggregate.yml"
    input:
        # mitotic montage data
        AGGREGATE_OUTPUTS["prepare_interphase_montage_data"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["generate_interphase_montage"],
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/generate_montage.py"


rule eval_aggregate:
    conda:
        "../envs/aggregate.yml"
    input:
        # cleaned data
        AGGREGATE_OUTPUTS["clean_transform_standardize"][0],
        # transformed data
        AGGREGATE_OUTPUTS["clean_transform_standardize"][1],
        # standardized data
        AGGREGATE_OUTPUTS["clean_transform_standardize"][2],
        # processed mitotic data
        AGGREGATE_OUTPUTS["process_mitotic_gene_data"],
        # processed interphase data
        AGGREGATE_OUTPUTS["process_interphase_gene_data"],
        # all processed gene data
        AGGREGATE_OUTPUTS["process_all_gene_data"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["eval_aggregate"],
    params:
        population_feature=config["aggregate"]["population_feature"],
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/eval_aggregate.py"


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
