"""Convert ND2 files to OME-Zarr format with multiscale pyramids.

Uses the unified save_image() I/O layer, which writes OME-Zarr via
write_image_omezarr(). This replaces the previous simple-zarr writer
(group with array at key "0") with a fully OME-NGFF compliant store
including multiscale pyramids and channel metadata.
"""

from lib.preprocess.preprocess import convert_to_array, get_data_config
from lib.shared.io import save_image

# Get data configuration from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Load and process image (same as TIFF conversion)
print(f"Loading {len(snakemake.input)} input files")
image_data = convert_to_array(
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
print(f"Loaded shape: {image_data.shape}, dtype: {image_data.dtype}")

# Get channel names from config for OME metadata
channel_names = data_config.get("channel_order")

# Write OME-Zarr with multiscale pyramids via unified I/O layer
print(f"Writing OME-Zarr to: {snakemake.output[0]}")
save_image(
    image_data,
    snakemake.output[0],
    channel_names=channel_names,
)

print(f"  OME-Zarr created successfully")
print(f"  Shape: {image_data.shape}")
print(f"  Dtype: {image_data.dtype}")
print(f"  Size: {image_data.nbytes / 1024**2:.1f} MB")
