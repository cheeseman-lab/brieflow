from lib.shared.target_utils import output_to_input


# Complete fast alignment process
rule fast_alignment:
    conda:
        "../envs/merge.yml"
    input:
        # metadata files with image locations
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
            (
                {}
                if config["merge"]["ph_metadata_channel"] is None
                else {"channel": config["merge"]["ph_metadata_channel"]}
            ),
            wildcards,
        ),
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["combine_metadata_sbs"],
            {
                "cycle": config["merge"]["sbs_metadata_cycle"],
                **(
                    {}
                    if config["merge"]["sbs_metadata_channel"] is None
                    else {"channel": config["merge"]["sbs_metadata_channel"]}
                ),
            },
            wildcards,
        ),
        PHENOTYPE_OUTPUTS["merge_phenotype_info"],
        SBS_OUTPUTS["combine_sbs_info"],
    output:
        MERGE_OUTPUTS_MAPPED["fast_alignment"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        initial_sites=config["merge"]["initial_sites"],
    script:
        "../scripts/merge/fast_alignment.py"


# Complete merge process
rule merge:
    conda:
        "../envs/merge.yml"
    input:
        # phenotype and sbs info files with cell locations
        PHENOTYPE_OUTPUTS["merge_phenotype_info"],
        SBS_OUTPUTS["combine_sbs_info"],
        # fast alignment data
        MERGE_OUTPUTS["fast_alignment"],
    output:
        MERGE_OUTPUTS_MAPPED["merge"],
    params:
        det_range=config["merge"]["det_range"],
        score=config["merge"]["score"],
        threshold=config["merge"]["threshold"],
    script:
        "../scripts/merge/merge.py"


# Format merge data
rule format_merge:
    conda:
        "../envs/merge.yml"
    input:
        # merge data
        MERGE_OUTPUTS["merge"],
        # cell information from SBS
        SBS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_OUTPUTS_MAPPED["format_merge"],
    script:
        "../scripts/merge/format_merge.py"


# Format merge data
rule eval_merge:
    conda:
        "../envs/merge.yml"
    input:
        # formatted merge data
        MERGE_OUTPUTS["format_merge"],
        # cell information from SBS
        SBS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_OUTPUTS_MAPPED["eval_merge"],
    script:
        "../scripts/merge/eval_merge.py"


# Clean merge data
rule clean_merge:
    conda:
        "../envs/merge.yml"
    input:
        # formatted merge data
        MERGE_OUTPUTS["format_merge"],
    output:
        MERGE_OUTPUTS_MAPPED["clean_merge"],
    params:
        channel_min_cutoff=0,
        misaligned_wells=None,
        misaligned_tiles=None,
    script:
        "../scripts/merge/clean_merge.py"


# Deduplicate merge data
rule deduplicate_merge:
    conda:
        "../envs/merge.yml"
    input:
        # cleaned merge data
        MERGE_OUTPUTS["clean_merge"][1],
        # cell information from SBS
        SBS_OUTPUTS["combine_cells"],
        # min phentoype information
        PHENOTYPE_OUTPUTS["merge_phenotype_cp"][1],
    output:
        MERGE_OUTPUTS_MAPPED["deduplicate_merge"],
    script:
        "../scripts/merge/deduplicate_merge.py"


# Final merge with all feature data
rule final_merge:
    conda:
        "../envs/merge.yml"
    input:
        # formatted merge data
        MERGE_OUTPUTS["deduplicate_merge"][1],
        # full phentoype information
        PHENOTYPE_OUTPUTS["merge_phenotype_cp"][0],
    output:
        MERGE_OUTPUTS_MAPPED["final_merge"],
    script:
        "../scripts/merge/final_merge.py"


# Rule for all merge processing steps
rule all_merge:
    input:
        MERGE_TARGETS_ALL,
