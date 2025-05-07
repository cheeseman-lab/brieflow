from lib.shared.target_utils import output_to_input, map_wildcard_outputs
from lib.shared.rule_utils import get_montage_inputs


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


rule generate_feature_table:
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
        AGGREGATE_OUTPUTS_MAPPED["generate_feature_table"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        perturbation_id_col=config["aggregate"]["perturbation_id_col"],
        control_key=config["aggregate"]["control_key"],
        batch_cols=config["aggregate"]["batch_cols"],
        batches=10,
    script:
        "../scripts/aggregate/generate_feature_table.py"


rule perturbation_score:
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
        AGGREGATE_OUTPUTS_MAPPED["perturbation_score"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        perturbation_id_col=config["aggregate"]["perturbation_id_col"],
        control_key=config["aggregate"]["control_key"],
        batch_cols=config["aggregate"]["batch_cols"],
        batches=10,
    script:
        "../scripts/aggregate/perturbation_score.py"


rule align:
    input:
        AGGREGATE_OUTPUTS_MAPPED["perturbation_score"],
    params:
        metadata_cols_fp=config["aggregate"]["metadata_cols_fp"],
        perturbation_name_col=config["aggregate"]["perturbation_name_col"],
        perturbation_id_col=config["aggregate"]["perturbation_id_col"],
        batch_cols=config["aggregate"]["batch_cols"],
        variance_or_ncomp=config["aggregate"]["variance_or_ncomp"],
        control_key=config["aggregate"]["control_key"],
        num_align_batches=config["aggregate"]["num_align_batches"],
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
        AGGREGATE_OUTPUTS_MAPPED["align"],
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


# MONTAGE CREATION
# NOTE: Montage creation happens dynamically
# We create a checkpoint once the montage data is prepared
# Then we initiate montage creation based on the checkpoint
# Then create a flag once montage creation is done


# Prepare montage data and create a checkpoint
checkpoint prepare_montage_data:
    input:
        lambda wildcards: output_to_input(
            AGGREGATE_OUTPUTS["filter"],
            wildcards={
                "cell_class": wildcards.cell_class,
                "channel_combo": aggregate_wildcard_combos["channel_combo"].unique()[
                    0
                ],
            },
            expansion_values=["plate", "well"],
            metadata_combos=aggregate_wildcard_combos,
        ),
    output:
        directory(MONTAGE_OUTPUTS["montage_data_dir"]),
    params:
        root_fp=config["all"]["root_fp"],
    script:
        "../scripts/aggregate/prepare_montage_data.py"


# Generate montage
rule generate_montage:
    input:
        MONTAGE_OUTPUTS["montage_data"],
    output:
        expand(
            str(MONTAGE_OUTPUTS["montage"]),
            cell_class="{cell_class}",
            gene="{gene}",
            sgrna="{sgrna}",
            channel=config["phenotype"]["channel_names"],
        )
        + [
            str(MONTAGE_OUTPUTS["montage_overlay"]).format(
                cell_class="{cell_class}", gene="{gene}", sgrna="{sgrna}"
            )
        ],
    params:
        channels=config["phenotype"]["channel_names"],
    script:
        "../scripts/aggregate/generate_montage.py"


# Initiate montage creation based on checkpoint
# Create a flag to indicate montage creation is done
rule initiate_montage:
    input:
        lambda wildcards: get_montage_inputs(
            checkpoints.prepare_montage_data,
            MONTAGE_OUTPUTS["montage"],
            MONTAGE_OUTPUTS["montage_overlay"],
            config["phenotype"]["channel_names"],
            wildcards.cell_class,
        ),
    output:
        touch(MONTAGE_OUTPUTS["montage_flag"]),


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
        # MONTAGE_TARGETS_ALL,
