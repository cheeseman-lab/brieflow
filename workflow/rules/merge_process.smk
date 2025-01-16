from lib.shared.target_utils import output_to_input

# Complete fast alignment process
rule fast_alignment:
    conda:
        "../envs/merge_process.yml"
    input:
        # metadata file with image locations
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
            {"well": PHENOTYPE_WELLS},
            wildcards,
        ),
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_sbs"],
            {"well": SBS_WELLS, "cycle": config["merge_process"]["sbs_metdata_cycle"]},
            wildcards,
        ),
        # phenotype and info files with cell locations
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_info"],
        SBS_PROCESS_OUTPUTS["combine_sbs_info"],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["fast_alignment"],
    script:
        "../scripts/merge_process/fast_alignment.py"
