"""Convert microscopy image files to the configured output format (TIFF or OME-Zarr).

Uses the unified save_image() I/O layer which dispatches based on output path suffix.
"""

from lib.preprocess.preprocess import convert_to_array, get_data_config
from lib.shared.image_io import (
    DEFAULT_MAX_LEVELS,
    DEFAULT_ZARR_COMPRESSION,
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

# Save in the format determined by output path extension.
# Pyramid depth and compression are zarr-only; ignored for TIFF output.
all_config = snakemake.config.get("all", {})
save_image(
    image_array,
    snakemake.output[0],
    channel_names=channel_names,
    max_levels=all_config.get("zarr_max_levels", DEFAULT_MAX_LEVELS),
    compression=all_config.get("zarr_compression", DEFAULT_ZARR_COMPRESSION),
)
