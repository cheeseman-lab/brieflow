"""Convert microscopy image files to the configured output format (TIFF or OME-Zarr).

Uses the unified save_image() I/O layer which dispatches based on output path suffix.
A second output, when the rule declares one, receives the tile's metadata table read
from the same input files.
"""

from lib.preprocess.preprocess import (
    convert_to_array,
    extract_metadata,
    get_data_config,
)
from lib.shared.image_io import (
    CONVERT_ZARR_COMPRESSION,
    DEFAULT_MAX_LEVELS,
    save_image,
)

# Get data configuration from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Convert the input files to an array
image_array = convert_to_array(
    snakemake.input,
    data_format=data_config["data_format"],
    data_organization=data_config["image_data_organization"],
    position=snakemake.params.tile
    if data_config["image_data_organization"] == "well"
    else None,
    channel_order_flip=data_config["channel_order_flip"],
    n_z_planes=data_config.get("n_z_planes"),
    verbose=False,
)

# Get channel names from config for OME metadata (used by zarr, ignored by tiff)
channel_names = data_config.get("channel_order")

# Format follows the output extension; pyramid depth and codec apply to zarr only.
all_config = snakemake.config.get("all", {})
save_image(
    image_array,
    snakemake.output[0],
    channel_names=channel_names,
    max_levels=all_config.get("zarr_max_levels", DEFAULT_MAX_LEVELS),
    compression=CONVERT_ZARR_COMPRESSION,
)

# Write the tile's metadata table when this rule also stands in for extract_metadata
if len(snakemake.output) > 1:
    metadata_df = extract_metadata(
        list(snakemake.input),
        plate=snakemake.params.plate,
        well=snakemake.params.well,
        tile=snakemake.params.metadata_tile,
        cycle=getattr(snakemake.params, "cycle", None),
        data_format=data_config["data_format"],
        data_organization=data_config["image_data_organization"],
        verbose=False,
    )
    metadata_df.to_csv(snakemake.output[1], index=False, sep="\t")
