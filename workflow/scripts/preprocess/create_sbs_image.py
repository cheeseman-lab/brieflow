"""Create the empty per-tile SBS images of one well, with sequencing cycles on C.

Reads the first tile of each cycle for its stain count, tile shape and data type,
then creates every tile's image; the convert_sbs jobs fill in one cycle each.
"""

from pathlib import Path

from lib.preprocess.preprocess import convert_to_array, get_data_config
from lib.shared.image_io import (
    CONVERT_ZARR_COMPRESSION,
    DEFAULT_MAX_LEVELS,
    create_cycle_stacked_image,
)

data_config = get_data_config(
    "sbs", {"preprocess": snakemake.config.get("preprocess", {})}
)
cycles = snakemake.params.cycles

n_stains, shapes, dtypes = [], set(), set()
for cycle in cycles:
    image = convert_to_array(
        snakemake.input[f"cycle_{cycle}"],
        data_format=data_config["data_format"],
        data_organization=data_config["image_data_organization"],
        position=snakemake.params.tile
        if data_config["image_data_organization"] == "well"
        else None,
        channel_order_flip=data_config["channel_order_flip"],
        n_z_planes=data_config.get("n_z_planes"),
        verbose=False,
    )
    n_stains.append(1 if image.ndim == 2 else image.shape[0])
    shapes.add(tuple(image.shape[-2:]))
    dtypes.add(image.dtype)

if len(shapes) != 1 or len(dtypes) != 1:
    raise ValueError(
        f"SBS cycles of one well differ in tile shape {sorted(shapes)} or data type "
        f"{sorted(map(str, dtypes))}; they cannot share one image per tile"
    )

shape, dtype = shapes.pop(), dtypes.pop()
max_levels = snakemake.config.get("all", {}).get("zarr_max_levels", DEFAULT_MAX_LEVELS)
for image_path in snakemake.params.image_paths:
    create_cycle_stacked_image(
        image_path,
        cycles,
        n_stains,
        shape,
        dtype,
        max_levels=max_levels,
        compression=CONVERT_ZARR_COMPRESSION,
    )

Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
Path(snakemake.output[0]).touch()
