from lib.shared.target_utils import output_to_input, map_wildcard_outputs


# Create datasets with cell classes and channel combos
rule split_datasets:
    input:
        # final merge data
        ancient(MERGE_OUTPUTS["final_merge"]),
    output:
        map_wildcard_outputs(
            aggregate_wildcard_combos,
            AGGREGATE_OUTPUTS["split_datasets"][0],
            ["cell_class", "channel_combo"],
        ),
    params:
        all_channels=config["phenotype"]["channel_names"],
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        classifier_path=config["aggregate"]["classifier_path"],
        cell_classes=aggregate_wildcard_combos["cell_class"].unique(),
        channel_combos=aggregate_wildcard_combos["channel_combo"].unique(),
    script:
        "../scripts/aggregate/split_datasets.py"


rule filter:
    input:
        AGGREGATE_OUTPUTS_MAPPED["split_datasets"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["filter"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        filter_queries=config["aggregate"]["filter_queries"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        drop_cols_threshold=config["aggregate"]["drop_cols_threshold"],
        drop_rows_threshold=config["aggregate"]["drop_rows_threshold"],
        impute=config["aggregate"]["impute"],
        channel_names=config["phenotype"]["channel_names"],
        contamination=config["aggregate"]["contamination"],
    script:
        "../scripts/aggregate/filter.py"


rule align:
    input:
        filtered_paths=lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS_MAPPED["filter"],
            wildcards={
                "cell_class": wildcards.cell_class,
                "channel_combo": wildcards.channel_combo,
            },
            expansion_values=["plate", "well"],
            metadata_combos=aggregate_wildcard_combos,
        ),
    output:
        AGGREGATE_OUTPUTS_MAPPED["align"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        perturbation_id_col=config["aggregate"]["perturbation_id_col"],
        batch_cols=config["aggregate"]["batch_cols"],
        variance_or_ncomp=config["aggregate"]["variance_or_ncomp"],
        control_key=config["aggregate"]["control_key"],
    script:
        "../scripts/aggregate/align.py"


rule aggregate:
    input:
        AGGREGATE_OUTPUTS_MAPPED["align"],
    output:
        AGGREGATE_OUTPUTS_MAPPED["aggregate"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        agg_method=config["aggregate"]["agg_method"],
    script:
        "../scripts/aggregate/aggregate.py"


rule eval_aggregate:
    input:
        # aggregated gene data
        AGGREGATE_OUTPUTS_MAPPED["aggregate"],
        # class merge data
        split_datasets_paths=lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS_MAPPED["split_datasets"],
            wildcards={
                "cell_class": wildcards.cell_class,
                "channel_combo": wildcards.channel_combo,
            },
            expansion_values=["plate", "well"],
            metadata_combos=aggregate_wildcard_combos,
        ),
    output:
        AGGREGATE_OUTPUTS_MAPPED["eval_aggregate"],
    script:
        "../scripts/aggregate/eval_aggregate.py"


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
