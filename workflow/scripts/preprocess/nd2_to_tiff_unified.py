from tifffile import imwrite
from lib.preprocess.preprocess import convert_to_array_unified, get_data_config

# Get data configuration from rule name
rule_name = snakemake.rule  # Will be "convert_sbs" or "convert_phenotype"
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(image_type, {"preprocess": snakemake.config.get("preprocess", {})})

# Convert the files to array using unified function
image_array = convert_to_array_unified(
    snakemake.input,
    data_format=data_config["data_format"],
    data_organization=data_config["data_organization"],
    position=snakemake.params.tile if data_config["data_organization"] == "well" else None,
    channel_order_flip=data_config["channel_order_flip"],
    verbose=False
)

# Save TIFF image array
imwrite(snakemake.output[0], image_array)
