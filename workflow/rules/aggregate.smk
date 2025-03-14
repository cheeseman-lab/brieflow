from lib.shared.target_utils import output_to_input, get_montage_inputs


# Split cells by classes to create datasets
rule create_datasets:
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
        AGGREGATE_OUTPUTS_MAPPED["create_datasets"],
    params:
        classifier_path=config["aggregate"]["classifier_path"],
    script:
        "../scripts/aggregate/create_datasets.py"


# Rule for all aggregate processing steps
rule all_aggregate:
    input:
        AGGREGATE_TARGETS_ALL,
