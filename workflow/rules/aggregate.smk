from lib.shared.target_utils import output_to_input, get_montage_inputs


# Clean, transform, and standardize merged data
rule clean_transform_standardize:
    conda:
        "../envs/aggregate.yml"
    input:
        # final merge data
        ancient(MERGE_OUTPUTS["final_merge"]),
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
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["split_phases"][0],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
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
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["split_phases"][1],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
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
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["clean_transform_standardize"][2],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
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


rule eval_aggregate:
    conda:
        "../envs/aggregate.yml"
    input:
        # cleaned data
        cleaned_data_paths=lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["clean_transform_standardize"][0],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
        # transformed data
        transformed_data_paths=lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["clean_transform_standardize"][1],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
        # standardized data
        standardized_data_paths=lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["clean_transform_standardize"][2],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
        # processed mitotic data
        mitotic_gene_data=AGGREGATE_OUTPUTS["process_mitotic_gene_data"],
        # processed interphase data
        interphase_gene_data=AGGREGATE_OUTPUTS["process_interphase_gene_data"],
        # all processed gene data
        all_gene_data=AGGREGATE_OUTPUTS["process_all_gene_data"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["eval_aggregate"],
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/eval_aggregate.py"


# MONTAGE CREATION
# NOTE: Montage creation happens dynamically
# We create a checkpoint once the montage data is prepared
# Then we initiate montage creation based on the checkpoint
# Then create a flag once montage creation is done


# Prepare montage data and create a checkpoint
checkpoint prepare_mitotic_montage_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # mitotic standardized data
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["split_phases"][0],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
    output:
        directory(MONTAGE_OUTPUTS["mitotic_montage_data_dir"]),
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate/prepare_montage_data.py"


# Generate montage
rule generate_mitotic_montage:
    conda:
        "../envs/aggregate.yml"
    input:
        MONTAGE_OUTPUTS["mitotic_montage_data"],
    output:
        expand(
            str(MONTAGE_OUTPUTS["mitotic_montage"]),
            gene="{gene}",
            sgrna="{sgrna}",
            channel=config["phenotype"]["channel_names"],
        ),
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/generate_montage.py"


# Initate montage creation based on checkpoint
# Create a flag to indicate montage creation is done
rule initiate_mitotic_montage:
    input:
        lambda wildcards: get_montage_inputs(
            checkpoints.prepare_mitotic_montage_data,
            MONTAGE_OUTPUTS["mitotic_montage"],
            config["phenotype"]["channel_names"],
        ),
    output:
        touch(MONTAGE_OUTPUTS["mitotic_montage_flag"]),


# Prepare montage data and create a checkpoint
checkpoint prepare_interphase_montage_data:
    conda:
        "../envs/aggregate.yml"
    input:
        # interphase standardized data
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["split_phases"][1],
            wildcards=wildcards,
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
        ),
    output:
        directory(MONTAGE_OUTPUTS["interphase_montage_data_dir"]),
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate/prepare_montage_data.py"


# Generate montage
rule generate_interphase_montage:
    conda:
        "../envs/aggregate.yml"
    input:
        MONTAGE_OUTPUTS["interphase_montage_data"],
    output:
        expand(
            str(MONTAGE_OUTPUTS["interphase_montage"]),
            gene="{gene}",
            sgrna="{sgrna}",
            channel=config["phenotype"]["channel_names"],
        ),
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/generate_montage.py"


# Initate montage creation based on checkpoint
# Create a flag to indicate montage creation is done
rule initiate_interphase_montage:
    input:
        lambda wildcards: get_montage_inputs(
            checkpoints.prepare_interphase_montage_data,
            MONTAGE_OUTPUTS["interphase_montage"],
            config["phenotype"]["channel_names"],
        ),
    output:
        touch(MONTAGE_OUTPUTS["interphase_montage_flag"]),


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
