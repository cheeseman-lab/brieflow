from lib.preprocess.file_utils import get_sample_fps

from lib.shared.target_utils import output_to_input

OME_ZARR_CFG = config.get("preprocess", {}).get("ome_zarr", {})
OME_ZARR_COMPRESSOR = OME_ZARR_CFG.get("compression")
OME_ZARR_CHUNK = OME_ZARR_CFG.get("chunk_xy", 1024)
OME_ZARR_CHUNK_SHAPE = config.get("preprocess", {}).get(
    "omezarr_chunk_shape", [1, OME_ZARR_CHUNK, OME_ZARR_CHUNK]
)

# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            tile=wildcards.tile,
            cycle=wildcards.cycle,
            channel_order=config["preprocess"]["sbs_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_sbs"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        tile=lambda wildcards: wildcards.tile,
        cycle=lambda wildcards: wildcards.cycle,
    script:
        "../scripts/preprocess/extract_tile_metadata.py"


# Combine metadata for SBS images on well level
rule combine_metadata_sbs:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_sbs"],
            wildcards=wildcards,
            expansion_values=["tile", "cycle"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_sbs"],
    script:
        "../scripts/shared/combine_dfs.py"


# Extract metadata for phenotype images
rule extract_metadata_phenotype:
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["phenotype_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["extract_metadata_phenotype"],
    params:
        plate=lambda wildcards: wildcards.plate,
        well=lambda wildcards: wildcards.well,
        tile=lambda wildcards: wildcards.tile,
    script:
        "../scripts/preprocess/extract_tile_metadata.py"


# Comine metadata for phenotype images on well level
rule combine_metadata_phenotype:
    input:
        lambda wildcards: output_to_input(
            PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=phenotype_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["combine_metadata_phenotype"],
    script:
        "../scripts/shared/combine_dfs.py"


# Convert SBS ND2 files to TIFF
rule convert_sbs:
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            cycle=wildcards.cycle,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["sbs_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs"],
    params:
        channel_order_flip=config["preprocess"]["sbs_channel_order_flip"],
    script:
        "../scripts/preprocess/nd2_to_tiff.py"


rule convert_sbs_omezarr:
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            cycle=wildcards.cycle,
            tile=wildcards.tile,
            channel_order=config["preprocess"]["sbs_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_sbs_omezarr"],
    params:
        channel_order_flip=config["preprocess"]["sbs_channel_order_flip"],
        chunk_shape=OME_ZARR_CHUNK_SHAPE,
        coarsening_factor=config["preprocess"].get("omezarr_coarsening_factor", 2),
        max_levels=config["preprocess"].get("omezarr_max_levels"),
        compressor=OME_ZARR_COMPRESSOR,
        channel_labels=config["sbs"]["channel_names"],
    script:
        "../scripts/preprocess/nd2_to_omezarr.py"


# Convert phenotype ND2 files to TIFF
rule convert_phenotype:
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            tile=wildcards.tile,
            round_order=config["preprocess"]["phenotype_round_order"],
            channel_order=config["preprocess"]["phenotype_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_phenotype"],
    params:
        channel_order_flip=config["preprocess"]["phenotype_channel_order_flip"],
    script:
        "../scripts/preprocess/nd2_to_tiff.py"


rule convert_phenotype_omezarr:
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df,
            plate=wildcards.plate,
            well=wildcards.well,
            tile=wildcards.tile,
            round_order=config["preprocess"]["phenotype_round_order"],
            channel_order=config["preprocess"]["phenotype_channel_order"],
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["convert_phenotype_omezarr"],
    params:
        channel_order_flip=config["preprocess"]["phenotype_channel_order_flip"],
        chunk_shape=OME_ZARR_CHUNK_SHAPE,
        coarsening_factor=config["preprocess"].get("omezarr_coarsening_factor", 2),
        max_levels=config["preprocess"].get("omezarr_max_levels"),
        compressor=OME_ZARR_COMPRESSOR,
        channel_labels=config["phenotype"]["channel_names"],
    script:
        "../scripts/preprocess/nd2_to_omezarr.py"


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
OME_ZARR_CFG = config["preprocess"].get("ome_zarr", {}) if "preprocess" in config else {}
OME_ZARR_COMPRESSOR = OME_ZARR_CFG.get(
    "compression",
    {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2},
)
