"""Image stitching estimation functions for SBS and phenotype data.

This module provides coordinate-based stitching estimation for two types of microscopy data:
- SBS (Sequencing By Synthesis): Lower resolution tiles (1200x1200 pixels)
- Phenotype: Higher resolution tiles (2400x2400 pixels)

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
) -> Dict[str, Dict]:
    """Estimate stitching positions using coordinate-based approach.

    This function converts stage coordinates (in micrometers) to pixel coordinates
    for image stitching. It uses either metadata pixel size or fallback specifications
    to ensure accurate scaling.

    Args:
        metadata_df: DataFrame containing tile metadata with columns:
            - well: Well identifier
            - x_pos, y_pos: Stage coordinates in micrometers
            - tile: Tile identifier
            - pixel_size_x, pixel_size_y: Optional pixel sizes in μm/pixel
        well: Well identifier to process
        data_type: Data type ("sbs" or "phenotype") to determine specifications

    Returns:
        Dictionary containing:
            - total_translation: Dict mapping tile IDs to [y_pixel, x_pixel] positions
            - confidence: Dict with confidence scores for each position
    """
    # Data type specifications
    specs = {
        "sbs": {
            "tile_size": (1200, 1200),
            "field_of_view_um": 1560.0,
            "display_name": "SBS"
        },
        "phenotype": {
            "tile_size": (2400, 2400), 
            "field_of_view_um": 260.0,
            "display_name": "Phenotype"
        }
    }
    
    spec = specs[data_type]
    tile_size = spec["tile_size"]
    field_of_view_um = spec["field_of_view_um"]
    display_name = spec["display_name"]
    
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No {display_name} tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values

    print(f"Creating coordinate-based {display_name} stitch config for {len(tile_ids)} tiles")

    # Detect actual spacing between adjacent tiles
    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"{display_name} detected spacing: {actual_spacing:.1f} μm")
    print(f"{display_name} tile size: {tile_size} pixels")

    # Determine pixel scaling from metadata or use fallback values
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"{display_name} pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"{display_name} pixels per micron: {pixels_per_micron:.4f}")

        # Verify pixel_size_y matches pixel_size_x
        pixel_size_y = well_metadata["pixel_size_y"].iloc[0]
        if abs(pixel_size_um - pixel_size_y) > 1e-6:
            print(
                f"⚠️  Warning: pixel_size_x ({pixel_size_um:.6f}) != pixel_size_y ({pixel_size_y:.6f})"
            )
    else:
        print(
            "⚠️  pixel_size_x/y not found in metadata, falling back to calculated values"
        )
        # Fallback: use specs
        pixels_per_micron = tile_size[0] / field_of_view_um
        print(f"{display_name} fallback - field of view: {field_of_view_um} μm")
        print(f"{display_name} fallback - pixels per micron: {pixels_per_micron:.4f}")

    x_min, y_min = coords.min(axis=0)

    total_translation = {}
    confidence = {}

    for i, tile_id in enumerate(tile_ids):
        x_pos, y_pos = coords[i]

        # Convert stage coordinates to pixel coordinates
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)

        total_translation[f"{well}/{tile_id}"] = [pixel_y, pixel_x]

        # High confidence since using direct coordinates
        confidence[f"coord_{i}"] = [[pixel_y, pixel_x], [pixel_y, pixel_x], 0.9]

    print(f"Generated {len(total_translation)} {display_name} coordinate-based positions")

    # OPTIMIZED verification section - avoid O(n²) loops
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        # Calculate average pixel spacing more efficiently
        pixel_coords = np.array(
            [[y_shifts[i], x_shifts[i]] for i in range(len(y_shifts))]
        )
        pixel_distances = pdist(pixel_coords)

        if len(pixel_distances) > 0:
            # Calculate stage distances for the same pairs
            stage_distances = pdist(coords)

            # Calculate pixel/stage ratios for verification
            valid_ratios = (
                pixel_distances[stage_distances > 0]
                / stage_distances[stage_distances > 0]
            )

            if len(valid_ratios) > 0:
                avg_pixel_spacing = np.mean(valid_ratios)
                print(
                    f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
                )

                # Check for expected tile overlap using average spacing
                expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
                actual_avg_spacing = np.mean(pixel_distances)

                if actual_avg_spacing > 0:
                    overlap_percent = (
                        (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                    )
                    print(f"{display_name} tile overlap: {overlap_percent:.1f}%")
                    if overlap_percent < 0:
                        print(
                            "⚠️  Warning: Negative overlap detected - tiles may have gaps"
                        )
                    elif overlap_percent > 50:
                        print(
                            "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                        )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"{display_name} final image size: {final_size}")
    print(f"{display_name} memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}