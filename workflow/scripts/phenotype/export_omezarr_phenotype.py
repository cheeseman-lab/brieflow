import pandas as pd
from tifffile import imread
from lib.shared.omezarr_writer import write_image_omezarr, write_labels_omezarr

# Read input image (aligned phenotype)
image_data = imread(snakemake.input.image)

# Extract pixel size from metadata if available
pixel_size_um = None
if hasattr(snakemake.input, "metadata") and hasattr(snakemake.params, "tile"):
    try:
        # Read parquet metadata
        meta_df = pd.read_parquet(snakemake.input.metadata)

        # Filter for current tile
        tile_id = int(snakemake.params.tile)
        tile_meta = meta_df[meta_df["tile"] == tile_id]

        if not tile_meta.empty:
            px = (
                tile_meta["pixel_size_x"].iloc[0]
                if "pixel_size_x" in tile_meta.columns
                else None
            )
            py = (
                tile_meta["pixel_size_y"].iloc[0]
                if "pixel_size_y" in tile_meta.columns
                else None
            )
            pz = (
                tile_meta["pixel_size_z"].iloc[0]
                if "pixel_size_z" in tile_meta.columns
                else None
            )

            def _valid(v):
                return v is not None and pd.notna(v) and float(v) > 0

            px_u = float(px) if _valid(px) else None
            py_u = float(py) if _valid(py) else (px_u if px_u is not None else None)
            pz_u = float(pz) if _valid(pz) else None

            if px_u is not None or py_u is not None or pz_u is not None:
                pixel_size_um = {"x": px_u, "y": py_u, "z": pz_u}
    except Exception as e:
        print(f"Warning: Failed to extract pixel size from metadata: {e}")

# Write image to OME-Zarr
write_image_omezarr(
    image_data=image_data,
    out_path=str(snakemake.output[0]),
    axes=snakemake.params.axes,
    pixel_size_um=pixel_size_um,
)

# Read and write nuclei labels
if hasattr(snakemake.input, "nuclei"):
    nuclei_data = imread(snakemake.input.nuclei)
    write_labels_omezarr(
        label_data=nuclei_data,
        out_path=str(snakemake.output[0]),
        label_name="nuclei",
        axes="yx",
        pixel_size_um=pixel_size_um,
    )

# Read and write cells labels
if hasattr(snakemake.input, "cells"):
    cells_data = imread(snakemake.input.cells)
    write_labels_omezarr(
        label_data=cells_data,
        out_path=str(snakemake.output[0]),
        label_name="cells",
        axes="yx",
        pixel_size_um=pixel_size_um,
    )
