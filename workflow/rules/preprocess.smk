from lib.preprocess.file_utils import (
    SampleIndex,
    get_inputs_for_metadata_extraction,
    metadata_from_convert,
    get_convert_mem_mb,
    get_ic_mem_mb,
    estimate_tile_mb,
    get_tiles_per_file,
)
from lib.preprocess.preprocess import get_data_config, include_tile_in_input, get_expansion_values
from lib.shared.target_utils import output_to_input, outputs_to_targets

# Sample tables indexed once, so input lookups never rescan them
sbs_sample_index = SampleIndex(sbs_samples_df)
phenotype_sample_index = SampleIndex(phenotype_samples_df)
sbs_metadata_sample_index = SampleIndex(sbs_metadata_samples_df)
phenotype_metadata_sample_index = SampleIndex(phenotype_metadata_samples_df)

# preprocess.fewer_jobs (absent in older configs, so finished screens never rerun) folds metadata into convert
SBS_METADATA_IN_CONVERT = metadata_from_convert(
    "sbs", config, sbs_metadata_wildcard_combos, sbs_metadata_samples_df
)
PHENOTYPE_METADATA_IN_CONVERT = metadata_from_convert(
    "phenotype", config, phenotype_metadata_wildcard_combos, phenotype_metadata_samples_df
)

FEWER_JOBS = bool(config.get("preprocess", {}).get("fewer_jobs", False))
SAMPLE_FRACTION = config.get("preprocess", {}).get("sample_fraction", 1)


# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        unpack(lambda wildcards: get_inputs_for_metadata_extraction(
            "sbs", config, sbs_sample_index, sbs_metadata_sample_index, wildcards
        ))
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: _get_well(wildcards),
        tile=lambda wildcards: getattr(wildcards, 'tile', None),
        cycle=lambda wildcards: getattr(wildcards, 'cycle', None),
    resources:
        mem_mb=lambda wildcards, input: get_convert_mem_mb(
            input.size_mb,
            get_tiles_per_file(
                sbs_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards))["sample_fp"].iloc[0],
                include_tile_in_input("sbs", config),
            ),
        ),
        runtime="20m",
    group:
        "preprocess_sbs"
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
            "phenotype", config, phenotype_sample_index, phenotype_metadata_sample_index, wildcards
        ))
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: _get_well(wildcards),
        tile=lambda wildcards: getattr(wildcards, 'tile', None),
        round=lambda wildcards: getattr(wildcards, 'round', None),
    resources:
        mem_mb=lambda wildcards, input: get_convert_mem_mb(
            input.size_mb,
            get_tiles_per_file(
                phenotype_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards))["sample_fp"].iloc[0],
                include_tile_in_input("phenotype", config),
            ),
        ),
        runtime="20m",
    group:
        "preprocess_phenotype"
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


# Convert SBS image files to the configured format
rule convert_sbs:
    input:
        lambda wildcards: sbs_sample_index.get_sample_fps(
            plate=wildcards.plate,
            well=_get_well(wildcards),
            cycle=wildcards.cycle,
            tile=wildcards.tile if include_tile_in_input("sbs", config) else None,
            channel_order=config.get("preprocess", {}).get("sbs_channel_order"),
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs"],
    params:
        tile=lambda wildcards: int(wildcards.tile),
    resources:
        mem_mb=lambda wildcards, input: get_convert_mem_mb(
            input.size_mb,
            get_tiles_per_file(
                sbs_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards))["sample_fp"].iloc[0],
                include_tile_in_input("sbs", config),
            ),
        ),
        runtime="20m",
    group:
        "preprocess_sbs"
    script:
        "../scripts/preprocess/convert_image.py"

# Convert phenotype image files to the configured format
rule convert_phenotype:
    input:
        lambda wildcards: phenotype_sample_index.get_sample_fps(
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
    resources:
        mem_mb=lambda wildcards, input: get_convert_mem_mb(
            input.size_mb,
            get_tiles_per_file(
                phenotype_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards))["sample_fp"].iloc[0],
                include_tile_in_input("phenotype", config),
            ),
        ),
        runtime="20m",
    group:
        "preprocess_phenotype"
    script:
        "../scripts/preprocess/convert_image.py"

# Convert and write each tile's metadata in one job, preferred over convert and extract_metadata
if SBS_METADATA_IN_CONVERT:

    rule convert_sbs_with_metadata:
        input:
            lambda wildcards: sbs_sample_index.get_sample_fps(
                plate=wildcards.plate,
                well=_get_well(wildcards),
                cycle=wildcards.cycle,
                tile=wildcards.tile,
                channel_order=config.get("preprocess", {}).get("sbs_channel_order"),
            ),
        output:
            PREPROCESS_OUTPUTS_MAPPED["convert_sbs"]
            + PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
        resources:
            mem_mb=lambda wildcards, input: get_convert_mem_mb(input.size_mb),
            runtime="20m",
        group:
            "preprocess_sbs"
        script:
            "../scripts/preprocess/convert_image_with_metadata.py"

    ruleorder: convert_sbs_with_metadata > convert_sbs > extract_metadata_sbs


if PHENOTYPE_METADATA_IN_CONVERT:

    rule convert_phenotype_with_metadata:
        input:
            lambda wildcards: phenotype_sample_index.get_sample_fps(
                plate=wildcards.plate,
                well=_get_well(wildcards),
                tile=wildcards.tile,
                round_order=config.get("preprocess", {}).get("phenotype_round_order"),
                channel_order=config.get("preprocess", {}).get("phenotype_channel_order"),
            ),
        output:
            PREPROCESS_OUTPUTS_MAPPED["convert_phenotype"]
            + PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
        resources:
            mem_mb=lambda wildcards, input: get_convert_mem_mb(input.size_mb),
            runtime="20m",
        group:
            "preprocess_phenotype"
        script:
            "../scripts/preprocess/convert_image_with_metadata.py"

    ruleorder: convert_phenotype_with_metadata > convert_phenotype > extract_metadata_phenotype


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
    resources:
        mem_mb=lambda wildcards, input: get_ic_mem_mb(
            len(input),
            SAMPLE_FRACTION,
            estimate_tile_mb(
                sbs_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards), cycle=wildcards.cycle),
                include_tile_in_input("sbs", config),
            ),
        ),
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
    resources:
        mem_mb=lambda wildcards, input: get_ic_mem_mb(
            len(input),
            SAMPLE_FRACTION,
            estimate_tile_mb(
                phenotype_sample_index.subset(plate=wildcards.plate, well=_get_well(wildcards)),
                include_tile_in_input("phenotype", config),
            ),
        ),
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# Assemble HCS plate-level metadata for zarr stores (zarr mode only)
if IMG_FMT == "zarr":

    # With fewer_jobs, each modality's finalize waits only for its own images, combined metadata and IC fields
    FINALIZE_INPUTS_SBS = list(dict.fromkeys(outputs_to_targets(
        {k: v for k, v in PREPROCESS_OUTPUTS_SBS.items() if k != "extract_metadata_sbs"},
        sbs_wildcard_combos,
        PREPROCESS_OUTPUT_MAPPINGS,
    )))
    FINALIZE_INPUTS_PHENOTYPE = list(dict.fromkeys(outputs_to_targets(
        {k: v for k, v in PREPROCESS_OUTPUTS_PHENOTYPE.items() if k != "extract_metadata_phenotype"},
        phenotype_wildcard_combos,
        PREPROCESS_OUTPUT_MAPPINGS,
    )))

    rule finalize_hcs_preprocess_sbs:
        input:
            FINALIZE_INPUTS_SBS if FEWER_JOBS else PREPROCESS_TARGETS_ALL,
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
            FINALIZE_INPUTS_PHENOTYPE if FEWER_JOBS else PREPROCESS_TARGETS_ALL,
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

