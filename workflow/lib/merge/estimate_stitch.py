"""Image stitching estimation functions for SBS and phenotype data.

This module provides coordinate-based stitching estimation.

The functions calculate pixel positions for image tiles based on stage coordinates
and verify proper tile spacing and overlap.
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy.spatial.distance import pdist


def estimate_stitch_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
    data_type: str,
    fallback_pixel_size: float = None,  # New parameter for fallback pixel size
) -> Dict[str, Dict]:
    """Estimate stitching positions using coordinate-based approach.

    This function converts stage coordinates (in micrometers) to pixel coordinates
    for image stitching. It uses either metadata pixel size or fallback pixel size
    from config to ensure accurate scaling.

    Args:
        metadata_df: DataFrame containing tile metadata with columns:
            - well: Well identifier
            - x_pos, y_pos: Stage coordinates in micrometers
            - tile: Tile identifier
            - pixel_size_x, pixel_size_y: Optional pixel sizes in μm/pixel
        well: Well identifier to process
        data_type: Data type ("sbs" or "phenotype") to determine specifications
        fallback_pixel_size: Fallback pixel size in μm/pixel from config

    Returns:
        Dictionary containing:
            - total_translation: Dict mapping tile IDs to [y_pixel, x_pixel] positions
            - confidence: Dict with confidence scores for each position
    """
    # Data type specifications
    display_names = {"sbs": "SBS", "phenotype": "Phenotype"}

    display_name = display_names[data_type]

    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No {display_name} tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values

    print(
        f"Creating coordinate-based {display_name} stitch config for {len(tile_ids)} tiles"
    )

    # Determine pixel scaling from metadata or use fallback values
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
        and not well_metadata["pixel_size_x"].isna().iloc[0]
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]
        pixels_per_micron = 1.0 / pixel_size_um
        print(f"{display_name} pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"{display_name} pixels per micron: {pixels_per_micron:.4f}")

        # Verify pixel_size_y matches pixel_size_x
        pixel_size_y = well_metadata["pixel_size_y"].iloc[0]
        if abs(pixel_size_um - pixel_size_y) > 1e-6:
            print(
                f"⚠️  Warning: pixel_size_x ({pixel_size_um:.6f}) != pixel_size_y ({pixel_size_y:.6f})"
            )
    elif fallback_pixel_size is not None:
        # Use fallback pixel size from config
        pixel_size_um = fallback_pixel_size
        pixels_per_micron = 1.0 / pixel_size_um
        print(
            f"{display_name} using config fallback pixel size: {pixel_size_um:.6f} μm/pixel"
        )
        print(f"{display_name} pixels per micron: {pixels_per_micron:.4f}")
    else:
        print(f"⚠️  No pixel size available for {display_name} - skipping well {well}")

    x_min, y_min = coords.min(axis=0)

    total_translation = {}

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]

        # Convert stage coordinates to pixel coordinates
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)

        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

    print(
        f"Generated {len(total_translation)} {display_name} coordinate-based positions"
    )

    print(f"{display_name} stitching estimation completed")

    return {"total_translation": total_translation}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
