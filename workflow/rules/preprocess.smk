from lib.preprocess.file_utils import (
    get_first_tile_inputs,
    get_inputs_for_metadata_extraction,
    get_sample_fps,
    get_well_tiles_and_cycles,
)
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
        well=lambda wildcards: _get_well(wildcards),
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
        well=lambda wildcards: _get_well(wildcards),
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
        well=lambda wildcards: _get_well(wildcards),
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
        well=lambda wildcards: _get_well(wildcards),
    script:
        "../scripts/preprocess/combine_metadata.py"


# Create the per-tile SBS images (cycles on C) of one well before its cycles are converted (zarr only)
if IMG_FMT == "zarr":

    rule create_sbs_image:
        input:
            unpack(
                lambda wildcards: get_first_tile_inputs(
                    sbs_samples_df, sbs_wildcard_combos, wildcards, config
                )
            ),
        output:
            PREPROCESS_OUTPUTS_MAPPED["create_sbs_image"],
        params:
            tile=lambda wildcards: int(
                get_well_tiles_and_cycles(sbs_wildcard_combos, wildcards)[0][0]
            ),
            cycles=lambda wildcards: get_well_tiles_and_cycles(
                sbs_wildcard_combos, wildcards
            )[1],
            image_paths=lambda wildcards: [
                PREPROCESS_SBS_IMAGE.format(
                    plate=wildcards.plate, row=wildcards.row, col=wildcards.col, tile=tile
                )
                for tile in get_well_tiles_and_cycles(sbs_wildcard_combos, wildcards)[0]
            ],
        script:
            "../scripts/preprocess/create_sbs_image.py"


# Convert SBS image files to the configured format
rule convert_sbs:
    input:
        images=lambda wildcards: get_sample_fps(
            sbs_samples_df,
            plate=wildcards.plate,
            well=_get_well(wildcards),
            cycle=wildcards.cycle,
            tile=wildcards.tile if include_tile_in_input("sbs", config) else None,
            channel_order=config.get("preprocess", {}).get("sbs_channel_order"),
        ),
        created=lambda wildcards: (
            output_to_input(PREPROCESS_OUTPUTS["create_sbs_image"], wildcards=wildcards)
            if IMG_FMT == "zarr"
            else []
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs"],
    params:
        tile=lambda wildcards: int(wildcards.tile),
        image_path=lambda wildcards: (
            PREPROCESS_SBS_IMAGE.format(
                plate=wildcards.plate, row=wildcards.row, col=wildcards.col, tile=wildcards.tile
            )
            if IMG_FMT == "zarr"
            else None
        ),
    script:
        "../scripts/preprocess/convert_image.py"

# Convert phenotype image files to the configured format
rule convert_phenotype:
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            plate=wildcards.plate,
            well=_get_well(wildcards),
            tile=wildcards.tile if include_tile_in_input("phenotype", config) else None,
            round_order=config.get("preprocess", {}).get("phenotype_round_order"),
            channel_order=config.get("preprocess", {}).get("phenotype_channel_order"),
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_phenotype"],
    params:
        tile=lambda wildcards: int(wildcards.tile),
    script:
        "../scripts/preprocess/convert_image.py"

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
        sample_fraction=config.get("preprocess", {}).get("sample_fraction", 1),
        image_paths=lambda wildcards: (
            output_to_input(
                PREPROCESS_SBS_IMAGE,
                wildcards=wildcards,
                expansion_values=["tile"],
                metadata_combos=sbs_wildcard_combos,
            )
            if IMG_FMT == "zarr"
            else None
        ),
        cycle=lambda wildcards: wildcards.cycle if IMG_FMT == "zarr" else None,
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
        sample_fraction=config.get("preprocess", {}).get("sample_fraction", 1),
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# Assemble HCS plate-level metadata for zarr stores (zarr mode only)
if IMG_FMT == "zarr":

    rule finalize_hcs_preprocess_sbs:
        input:
            PREPROCESS_TARGETS_ALL,
        output:
            touch(str(PREPROCESS_FP / ".hcs_done_sbs")),
        params:
            plate_zarr_dirs=[str(PREPROCESS_FP / "sbs" / f"image_{p}.zarr")
                             for p in sorted(sbs_wildcard_combos["plate"].unique())],
            channels_metadata=config["preprocess"].get("sbs_channels_metadata", None),
        threads: 8
        script:
            "../scripts/shared/write_hcs_metadata.py"


    rule finalize_hcs_preprocess_phenotype:
        input:
            PREPROCESS_TARGETS_ALL,
        output:
            touch(str(PREPROCESS_FP / ".hcs_done_phenotype")),
        params:
            plate_zarr_dirs=[str(PREPROCESS_FP / "phenotype" / f"image_{p}.zarr")
                             for p in sorted(phenotype_wildcard_combos["plate"].unique())],
            channels_metadata=config["preprocess"].get("phenotype_channels_metadata", None),
        threads: 8
        script:
            "../scripts/shared/write_hcs_metadata.py"


# rule for all preprocessing steps
rule all_preprocess:
    input:
        PREPROCESS_TARGETS_ALL + (
            [str(PREPROCESS_FP / ".hcs_done_sbs"),
             str(PREPROCESS_FP / ".hcs_done_phenotype")] if IMG_FMT == "zarr" else []
        )

