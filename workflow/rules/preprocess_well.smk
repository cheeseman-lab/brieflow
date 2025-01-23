from lib.preprocess.file_utils import get_sample_fps
from lib.shared.target_utils import output_to_input

# Extract metadata for SBS images
rule extract_metadata_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            well=wildcards.well,
            cycle=wildcards.cycle,
            channel=wildcards.channel,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
    script:
        "../scripts/preprocess/extract_well_metadata.py"

# Extract metadata for phenotype images
rule extract_metadata_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            well=wildcards.well,
            channel=wildcards.channel,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
    script:
        "../scripts/preprocess/extract_well_metadata.py"

# rule for all preprocessing steps
rule all_preprocess:
    input:
        PREPROCESS_TARGETS_ALL,