"""Script to convert microscopy image files to TIFF format."""

from tifffile import imwrite
from lib.preprocess.preprocess import convert_to_array, get_data_config

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
    verbose=False,
)

# Save as TIFF file
imwrite(snakemake.output[0], image_array)
