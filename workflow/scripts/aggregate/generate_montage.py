import pandas as pd
import numpy as np
from pathlib import Path

from lib.aggregate.montage_utils import create_cell_montage, extract_cell_crops

# Read parameters
IMG_FMT = snakemake.params.get("img_fmt", "tiff")

# read cell data
montage_data = pd.read_csv(snakemake.input[0], sep="\t")

if IMG_FMT == "zarr":
    import zarr

    # Extract individual cell crops → write to examples.zarr
    examples_zarr_root = Path(snakemake.params.examples_zarr_root)
    gene = snakemake.wildcards.gene
    barcode = snakemake.wildcards.sgrna

    cell_crops = extract_cell_crops(
        montage_data,
        num_cells=snakemake.params.get("montage_num_cells", 30),
        cell_size=snakemake.params.get("montage_cell_size", 40),
    )

    # Write each crop as a zarr array under examples.zarr/{gene}/{barcode}/{idx}
    group_path = examples_zarr_root / gene / barcode
    group_path.mkdir(parents=True, exist_ok=True)

    for idx in range(len(cell_crops)):
        crop = cell_crops[idx]  # (C, H, W)
        arr_path = group_path / str(idx)
        z = zarr.open(str(arr_path), mode="w", shape=crop.shape, dtype=crop.dtype)
        z[:] = crop

    print(f"Wrote {len(cell_crops)} crops to {group_path}")

    # Touch output flag
    Path(snakemake.output[0]).touch()

else:
    from skimage.exposure import rescale_intensity
    from skimage.io import imsave
    import tifffile

    # create cell montage (existing behavior)
    montage = create_cell_montage(
        montage_data,
        snakemake.params.channels,
        cell_size=snakemake.params.get("montage_cell_size", 40),
        shape=tuple(snakemake.params.get("montage_shape", (3, 10))),
    )

    # save montages
    overlay = []
    for index, channel_montage in enumerate(montage.values()):
        overlay.append(channel_montage)
        montage_uint8 = rescale_intensity(
            channel_montage, in_range="image", out_range=(0, 255)
        ).astype(np.uint8)
        imsave(snakemake.output[index], montage_uint8)

    # save overlay stack as multi-channel TIFF
    overlay_stack = np.stack(overlay, axis=0)
    overlay_index = len(snakemake.output) - 1
    tifffile.imwrite(snakemake.output[overlay_index], overlay_stack)
