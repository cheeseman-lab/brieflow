from lib.preprocess.file_utils import get_sample_fps, get_inputs_for_metadata_extraction
from lib.preprocess.preprocess import get_data_config, include_tile_in_input, get_expansion_values
from lib.shared.target_utils import output_to_input


# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        unpack(lambda wildcards: get_inputs_for_metadata_extraction(
            "sbs", config, sbs_samples_df, sbs_metadata_samples_df, wildcards
        ))
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: getattr(wildcards, 'well', None),
        tile=lambda wildcards: getattr(wildcards, 'tile', None),
        cycle=lambda wildcards: getattr(wildcards, 'cycle', None),
    script:
        "../scripts/preprocess/extract_metadata.py"


# Combine metadata for SBS images on well level
rule combine_metadata_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_sbs"],
            wildcards=wildcards,
            expansion_values=get_expansion_values("sbs", config),
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_sbs"],
    script:
        "../scripts/shared/combine_dfs.py"


# Extract metadata for phenotype images
rule extract_metadata_phenotype:
    input:
        unpack(lambda wildcards: get_inputs_for_metadata_extraction(
            "phenotype", config, phenotype_samples_df, phenotype_metadata_samples_df, wildcards
        ))
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: getattr(wildcards, 'well', None),
        tile=lambda wildcards: getattr(wildcards, 'tile', None),
        round=lambda wildcards: getattr(wildcards, 'round', None),
    script:
        "../scripts/preprocess/extract_metadata.py"


# Combine metadata for phenotype images on well level
rule combine_metadata_phenotype:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
            wildcards=wildcards,
            expansion_values=get_expansion_values("phenotype", config),
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_phenotype"],
    script:
        "../scripts/shared/combine_dfs.py"


# Convert SBS image files to TIFF
rule convert_sbs:
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            cycle=wildcards.cycle,
            tile=wildcards.tile if include_tile_in_input("sbs", config) else None,
            channel_order=config["preprocess"]["sbs_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs"],
    params:
        tile=lambda wildcards: int(wildcards.tile),
    script:
        "../scripts/preprocess/image_to_tiff.py"


# Convert phenotype image files to TIFF
rule convert_phenotype:
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            tile=wildcards.tile if include_tile_in_input("phenotype", config) else None,
            round_order=config["preprocess"]["phenotype_round_order"],
            channel_order=config["preprocess"]["phenotype_channel_order"]
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_phenotype"],
    params:
        tile=lambda wildcards: int(wildcards.tile),
    script:
        "../scripts/preprocess/image_to_tiff.py"


# Calculate illumination correction function for SBS files
rule calculate_ic_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_sbs"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["calculate_ic_sbs"],
    params:
        threading=True,
        sample_fraction=config["preprocess"]["sample_fraction"],
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# Calculate illumination correction for phenotype files
rule calculate_ic_phenotype:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["convert_phenotype"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["calculate_ic_phenotype"],
    params:
        threading=True,
        sample_fraction=config["preprocess"]["sample_fraction"],
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# rule for all preprocessing steps
rule all_preprocess:
    input:
        PREPROCESS_TARGETS_ALL,