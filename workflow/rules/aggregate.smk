from lib.shared.target_utils import output_to_input, get_montage_inputs


# Split cells by classes
rule split_classes:
    input:
        # final merge data
        merge_data_paths=lambda wildcards: output_to_input(
            MERGE_OUTPUTS["final_merge"],
            wildcards={},
            expansion_values=["plate", "well"],
            metadata_combos=merge_wildcard_combos,
            ancient_output=True,
        ),
    output:
        AGGREGATE_OUTPUTS_MAPPED["split_classes"],
    params:
        cell_class=lambda wildcards: wildcards.cell_class,
        classifier_path=config["aggregate"]["classifier_path"],
        first_feature=config["aggregate"]["first_feature"],
    script:
        "../scripts/aggregate/split_classes.py"


rule align_filter_aggregate:
    input:
        AGGREGATE_OUTPUTS_MAPPED["split_classes"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["align_filter_aggregate"],
    params:
        first_feature=config["aggregate"]["first_feature"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        perturbation_multi_col=config["aggregate"]["perturbation_multi_col"],
        filter_single_pert=config["aggregate"]["filter_single_pert"],
        drop_cols_threshold=config["aggregate"]["drop_cols_threshold"],
        drop_rows_threshold=config["aggregate"]["drop_rows_threshold"],
        impute=config["aggregate"]["impute"],
        channel_names=config["phenotype"]["channel_names"],
        contamination=config["aggregate"]["contamination"],
        batch_cols=config["aggregate"]["batch_cols"],
        pc_count=config["aggregate"]["pc_count"],
        control_key=config["aggregate"]["control_key"],
        agg_method=config["aggregate"]["agg_method"],
    script:
        "../scripts/aggregate/align_filter_aggregate.py"


rule eval_aggregate:
    input:
        # class merge data
        AGGREGATE_OUTPUTS_MAPPED["split_classes"],
        # aggregated gene data
        AGGREGATE_OUTPUTS_MAPPED["align_filter_aggregate"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["eval_aggregate"],
    script:
        "../scripts/aggregate/eval_aggregate.py"


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
