import pandas as pd
from tifffile import imread
from lib.io.omezarr_writer import write_image_omezarr, write_labels_omezarr

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
            # Get pixel_size_x (assuming square pixels usually)
            if "pixel_size_x" in tile_meta.columns:
                px = tile_meta["pixel_size_x"].iloc[0]
                if pd.notna(px) and px > 0:
                    pixel_size_um = float(px)
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
