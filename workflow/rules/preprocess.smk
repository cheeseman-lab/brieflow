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
            tile=wildcards.tile,
            cycle=wildcards.cycle,
            channel_order=config["preprocess"]["sbs_channel_order"]
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
    params:
        tile=lambda wildcards: wildcards.tile,
    script:
        "../scripts/preprocess/extract_tile_metadata.py"


# Extract metadata for SBS images
rule combine_metadata_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_sbs"],
            {"tile": SBS_TILES},
            wildcards,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_sbs"],
    params:
        output_type="tsv",
    script:
        "../scripts/shared/combine_dfs.py"


# Extract metadata for phenotype images
rule extract_metadata_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            well=wildcards.well,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["phenotype_channel_order"]
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
    params:
        tile=lambda wildcards: wildcards.tile,
    script:
        "../scripts/preprocess/extract_tile_metadata.py"


# Extract metadata for SBS images
rule combine_metadata_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
            {"tile": PHENOTYPE_TILES},
            wildcards,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_phenotype"],
    params:
        output_type="tsv",
    script:
        "../scripts/shared/combine_dfs.py"


# Convert SBS ND2 files to TIFF
rule convert_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            well=wildcards.well,
            cycle=wildcards.cycle,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["sbs_channel_order"]
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs"],
    params:
        channel_order_flip=config["preprocess"]["sbs_channel_order_flip"],
    script:
        "../scripts/preprocess/nd2_to_tiff.py"


# Convert phenotype ND2 files to TIFF
rule convert_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            well=wildcards.well,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["phenotype_channel_order"]
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_phenotype"],
    params:
        channel_order_flip=config["preprocess"]["phenotype_channel_order_flip"],
    script:
        "../scripts/preprocess/nd2_to_tiff.py"


# Calculate illumination correction function for SBS files
rule calculate_ic_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_sbs"],
            {"tile": SBS_TILES},
            wildcards,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["calculate_ic_sbs"],
    params:
        threading=True,
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# Calculate illumination correction for phenotype files
rule calculate_ic_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_phenotype"],
            {"tile": PHENOTYPE_TILES},
            wildcards,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["calculate_ic_phenotype"],
    params:
        threading=True,
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# rule for all preprocessing steps
rule all_preprocess:
    input:
        PREPROCESS_TARGETS_ALL,
