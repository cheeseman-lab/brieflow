"""Convert one tile's image files and write the tile's metadata table from the same files.

Used instead of convert_image.py when preprocess.fewer_jobs is set and the input
is tile-organized ND2 or IMS read from its headers (see metadata_from_convert).
"""

from lib.preprocess.preprocess import (
    convert_to_array,
    extract_metadata,
    get_data_config,
)
from lib.shared.file_utils import get_well_from_wildcards
from lib.shared.image_io import (
    CONVERT_ZARR_COMPRESSION,
    DEFAULT_MAX_LEVELS,
    save_image,
)

image_type = "sbs" if "sbs" in snakemake.rule else "phenotype"
data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)
input_files = list(snakemake.input)

# Same conversion as convert_image.py for a tile-organized input
image_array = convert_to_array(
    input_files,
    data_format=data_config["data_format"],
    data_organization=data_config["image_data_organization"],
    channel_order_flip=data_config["channel_order_flip"],
    n_z_planes=data_config.get("n_z_planes"),
    verbose=False,
)
all_config = snakemake.config.get("all", {})
save_image(
    image_array,
    snakemake.output[0],
    channel_names=data_config.get("channel_order"),
    max_levels=all_config.get("zarr_max_levels", DEFAULT_MAX_LEVELS),
    compression=CONVERT_ZARR_COMPRESSION,
)

# Same metadata table, with the same arguments, as extract_metadata.py
wildcards = snakemake.wildcards
metadata_df = extract_metadata(
    input_files,
    plate=wildcards.plate,
    well=get_well_from_wildcards(wildcards),
    tile=wildcards.tile,
    cycle=getattr(wildcards, "cycle", None),
    data_format=data_config["data_format"],
    data_organization=data_config["image_data_organization"],
    verbose=False,
)
metadata_df.to_csv(snakemake.output[1], index=False, sep="\t")
