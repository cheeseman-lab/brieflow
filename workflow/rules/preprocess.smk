from lib.preprocess.file_utils import get_sample_fps

from lib.shared.target_utils import output_to_input

OME_ZARR_CFG = config.get("preprocess", {}).get("ome_zarr", {})
OME_ZARR_COMPRESSOR = OME_ZARR_CFG.get("compression")
OME_ZARR_CHUNK = OME_ZARR_CFG.get("chunk_xy", 1024)
OME_ZARR_CHUNK_SHAPE = config.get("preprocess", {}).get(
    "omezarr_chunk_shape", [1, OME_ZARR_CHUNK, OME_ZARR_CHUNK]
)

# Determine whether to use OME-Zarr or TIFF for intermediate steps
output_formats = config.get("preprocess", {}).get("output_formats", ["zarr"])
if isinstance(output_formats, str):
    output_formats = [output_formats]

ENABLE_ZARR = "zarr" in output_formats
ENABLE_TIFF = "tiff" in output_formats

# Determine downstream input format
# Default to TIFF if enabled, otherwise Zarr
default_downstream = "tiff" if ENABLE_TIFF else "zarr"
downstream_format = config.get("preprocess", {}).get("downstream_input_format", default_downstream)

if downstream_format == "zarr" and not ENABLE_ZARR:
    raise ValueError("Downstream format is Zarr but Zarr output is not enabled.")
if downstream_format == "tiff" and not ENABLE_TIFF:
    raise ValueError("Downstream format is TIFF but TIFF output is not enabled.")

# Set keys for downstream use
CONVERT_SBS_KEY = "convert_sbs_omezarr" if downstream_format == "zarr" else "convert_sbs"
CONVERT_PHENOTYPE_KEY = "convert_phenotype_omezarr" if downstream_format == "zarr" else "convert_phenotype"

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


if ENABLE_TIFF:
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

if ENABLE_ZARR:
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
            PREPROCESS_OUTPUTS[CONVERT_SBS_KEY],
            wildcards=wildcards,
            expansion_values=["tile"],
            metadata_combos=sbs_wildcard_combos,
        ),
    output:
        PREPROCESS_OUTPUTS_MAPPED["calculate_ic_sbs"],
    params:
        threading=False,
        sample_fraction=config["preprocess"]["sample_fraction"],
        pixel_size_z=config["pixel_size_z"],
        pixel_size_y=config["pixel_size_y"],
        pixel_size_x=config["pixel_size_x"],
        channel_names=config["sbs"]["channel_names"],
        smooth=config["preprocess"].get("ic_smooth_factor"),
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
        threading=False,
        sample_fraction=config["preprocess"]["sample_fraction"],
        pixel_size_z=config["pixel_size_z"],
        pixel_size_y=config["pixel_size_y"],
        pixel_size_x=config["pixel_size_x"],
        channel_names=config["phenotype"]["channel_names"],
        smooth=config["preprocess"].get("ic_smooth_factor"),
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
