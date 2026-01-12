"""Script to convert microscopy image files directly to OME-Zarr format."""

from lib.preprocess.preprocess import convert_to_array, get_data_config
from lib.shared.omezarr_writer import write_image_omezarr

# Get data configuration from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Convert the files to array using main function
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

# Determine axes based on image dimensionality
if image_array.ndim == 2:
    axes = "yx"
elif image_array.ndim == 3:
    axes = "cyx"
elif image_array.ndim == 4:
    axes = "czyx"
else:
    raise ValueError(f"Unsupported image dimensionality: {image_array.ndim}")

# Get channel names if available
channel_names = data_config.get("channel_order")

# Save directly as OME-Zarr
write_image_omezarr(
    image_data=image_array,
    out_path=str(snakemake.output[0]),
    channel_names=channel_names,
    axes=axes,
    pixel_size_um=None,  # TODO: extract from metadata if available
    coarsening_factor=2,
    max_levels=5,
)
