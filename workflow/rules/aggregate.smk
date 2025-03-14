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


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
