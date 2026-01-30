from lib.preprocess.file_utils import get_sample_fps, get_inputs_for_metadata_extraction
from lib.preprocess.preprocess import get_data_config, include_tile_in_input, get_expansion_values
from lib.shared.target_utils import output_to_input


# CONVERT_SBS_KEY, CONVERT_PHENOTYPE_KEY, USE_OME_ZARR are defined in targets/preprocess.smk

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


# Combine metadata for SBS images
rule combine_metadata_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_sbs"],
            wildcards=wildcards,
            expansion_values=get_expansion_values("sbs", config, sbs_metadata_wildcard_combos),
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_sbs"],
    params:
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/preprocess/combine_metadata.py"


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


# Combine metadata for phenotype images
rule combine_metadata_phenotype:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
            wildcards=wildcards,
            expansion_values=get_expansion_values("phenotype", config, phenotype_metadata_wildcard_combos),
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_phenotype"],
    params:
        well=lambda wildcards: wildcards.well,
    script:
        "../scripts/preprocess/combine_metadata.py"


# Define conversion rules based on output format
if "convert_sbs" in PREPROCESS_OUTPUTS_MAPPED:
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

if "convert_phenotype" in PREPROCESS_OUTPUTS_MAPPED:
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

# Standard Zarr conversion (for downstream processing)
if "convert_sbs_zarr" in PREPROCESS_OUTPUTS_MAPPED:
    rule convert_sbs_zarr:
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
            PREPROCESS_OUTPUTS_MAPPED["convert_sbs_zarr"],
        params:
            tile=lambda wildcards: int(wildcards.tile),
        script:
            "../scripts/preprocess/nd2_to_zarr.py"

if "convert_phenotype_zarr" in PREPROCESS_OUTPUTS_MAPPED:
    rule convert_phenotype_zarr:
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
            PREPROCESS_OUTPUTS_MAPPED["convert_phenotype_zarr"],
        params:
            tile=lambda wildcards: int(wildcards.tile),
        script:
            "../scripts/preprocess/nd2_to_zarr.py"

# Calculate illumination correction function for SBS files
rule calculate_ic_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS[CONVERT_SBS_KEY],
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
            PREPROCESS_OUTPUTS[CONVERT_PHENOTYPE_KEY],
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


# Assemble HCS plate-level zarr stores from per-tile outputs (zarr mode only)
if ENABLE_ZARR:
    rule finalize_hcs_preprocess:
        input:
            PREPROCESS_TARGETS_ALL,
        output:
            touch(str(PREPROCESS_FP / ".hcs_done")),
        params:
            images_dir=str(PREPROCESS_FP / "images"),
            hcs_dir=str(PREPROCESS_FP / "hcs"),
        script:
            "../scripts/shared/write_hcs_metadata.py"


# rule for all preprocessing steps
rule all_preprocess:
    input:
        PREPROCESS_TARGETS_ALL,