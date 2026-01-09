import numpy as np
import pandas as pd
from tifffile import imread
from lib.io.omezarr_writer import write_image_omezarr

# Read input TIFF
image_data = imread(snakemake.input.image)

# Ensure correct dimensions (C, Y, X)
# tifffile might return (Y, X) for 2D, or (C, Y, X) for 3D
if image_data.ndim == 2:
    # Add channel dim
    image_data = image_data[np.newaxis, :, :]

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
                    # Apply upsample factor if present (default 1.0)
                    upsample = float(snakemake.params.get("upsample_factor", 1.0))
                    pixel_size_um = float(px) / upsample
    except Exception as e:
        print(f"Warning: Failed to extract pixel size from metadata: {e}")

# Write OME-Zarr
write_image_omezarr(
    image_data=image_data,
    out_path=snakemake.output[0],
    axes=snakemake.params.axes,
    channel_names=snakemake.params.get("channel_names"),
    pixel_size_um=pixel_size_um,
)
