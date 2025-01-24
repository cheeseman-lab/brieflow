from lib.shared.target_utils import output_to_input


# Complete fast alignment process
rule fast_alignment:
    conda:
        "../envs/merge_process.yml"
    input:
        # metadata files with image locations
        PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_sbs"],
            {"cycle": config["merge_process"]["sbs_metdata_cycle"]},
            wildcards,
        ),
        # phenotype and sbs info files with cell locations
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_info"],
        SBS_PROCESS_OUTPUTS["combine_sbs_info"],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["fast_alignment"],
    params:
        det_range=config["merge_process"]["det_range"],
        score=config["merge_process"]["score"],
        initial_sites=config["merge_process"]["initial_sites"],
    script:
        "../scripts/merge_process/fast_alignment.py"


# Complete merge process
rule merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # phenotype and sbs info files with cell locations
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_info"],
        SBS_PROCESS_OUTPUTS["combine_sbs_info"],
        # fast alignment data
        MERGE_PROCESS_OUTPUTS["fast_alignment"],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["merge"],
    params:
        det_range=config["merge_process"]["det_range"],
        score=config["merge_process"]["score"],
        threshold=config["merge_process"]["threshold"],
    script:
        "../scripts/merge_process/merge.py"


# Format merge data
rule format_merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # merge data
        MERGE_PROCESS_OUTPUTS["merge"],
        # cell information from SBS
        SBS_PROCESS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["format_merge"],
    script:
        "../scripts/merge_process/format_merge.py"


# Format merge data
rule eval_merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # formatted merge data
        MERGE_PROCESS_OUTPUTS["format_merge"],
        # cell information from SBS
        SBS_PROCESS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["eval_merge"],
    script:
        "../scripts/merge_process/eval_merge.py"


# Clean merge data
rule clean_merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # formatted merge data
        MERGE_PROCESS_OUTPUTS["format_merge"],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["clean_merge"],
    params:
        channel_min_cutoff=0,
        misaligned_wells=None,
        misaligned_tiles=None,
    script:
        "../scripts/merge_process/clean_merge.py"


# Deduplicate merge data
rule deduplicate_merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # cleaned merge data
        MERGE_PROCESS_OUTPUTS["clean_merge"],
        # cell information from SBS
        SBS_PROCESS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["deduplicate_merge"],
    script:
        "../scripts/merge_process/deduplicate_merge.py"


# Final merge with all feature data
rule final_merge:
    conda:
        "../envs/merge_process.yml"
    input:
        # formatted merge data
        MERGE_PROCESS_OUTPUTS["deduplicate_merge"],
        # full phentoype information
        PHENOTYPE_PROCESS_OUTPUTS["merge_phenotype_cp"][0],
    output:
        MERGE_PROCESS_OUTPUTS_MAPPED["final_merge"],
    script:
        "../scripts/merge_process/final_merge.py"


# Rule for all merge processing steps
rule all_merge_process:
    input:
        MERGE_PROCESS_TARGETS_ALL,
