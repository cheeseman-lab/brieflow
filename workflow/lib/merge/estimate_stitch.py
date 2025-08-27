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


def estimate_stitch_sbs_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
) -> Dict[str, Dict]:
    """Estimate stitching positions for SBS tiles using coordinate-based approach.
    
    This function converts stage coordinates (in micrometers) to pixel coordinates
    for image stitching. It uses either metadata pixel size or fallback SBS 
    specifications to ensure accurate scaling.
    
    Args:
        metadata_df: DataFrame containing tile metadata with columns:
            - well: Well identifier
            - x_pos, y_pos: Stage coordinates in micrometers
            - tile: Tile identifier
            - pixel_size_x, pixel_size_y: Optional pixel sizes in μm/pixel
        well: Well identifier to process
        
    Returns:
        Dictionary containing:
            - total_translation: Dict mapping tile IDs to [y_pixel, x_pixel] positions
            - confidence: Dict with confidence scores for each position
            
    Note:
        SBS specifications: 1560 μm field of view, 1200x1200 pixel tiles
    """
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No SBS tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values
    tile_size = (1200, 1200)  # SBS tile size in pixels

    print(f"Creating coordinate-based SBS stitch config for {len(tile_ids)} tiles")

    # Detect actual spacing between adjacent tiles
    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"SBS detected spacing: {actual_spacing:.1f} μm")
    print(f"SBS tile size: {tile_size} pixels")

    # Determine pixel scaling from metadata or use fallback values
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"SBS pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"SBS pixels per micron: {pixels_per_micron:.4f}")

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
        # Fallback: SBS specs: 1560 μm field of view, 1200 pixels -> 0.7692 pixels/μm
        sbs_field_of_view_um = 1560.0  # μm
        pixels_per_micron = tile_size[0] / sbs_field_of_view_um
        print(f"SBS fallback - field of view: {sbs_field_of_view_um} μm")
        print(f"SBS fallback - pixels per micron: {pixels_per_micron:.4f}")

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

    print(f"Generated {len(total_translation)} SBS coordinate-based positions")

    # Verify output size and spacing
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Check for expected tile overlap
            expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"SBS tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"SBS final image size: {final_size}")
    print(f"SBS memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}


def estimate_stitch_phenotype_coordinate_based(
    metadata_df: pd.DataFrame,
    well: str,
) -> Dict[str, Dict]:
    """Estimate stitching positions for phenotype tiles using coordinate-based approach.
    
    This function converts stage coordinates (in micrometers) to pixel coordinates
    for image stitching. It uses either metadata pixel size or fallback phenotype
    specifications to ensure accurate scaling.
    
    Args:
        metadata_df: DataFrame containing tile metadata with columns:
            - well: Well identifier
            - x_pos, y_pos: Stage coordinates in micrometers
            - tile: Tile identifier
            - pixel_size_x, pixel_size_y: Optional pixel sizes in μm/pixel
        well: Well identifier to process
        
    Returns:
        Dictionary containing:
            - total_translation: Dict mapping tile IDs to [y_pixel, x_pixel] positions
            - confidence: Dict with confidence scores for each position
            
    Note:
        Phenotype specifications: 260 μm field of view, 2400x2400 pixel tiles
    """
    well_metadata = metadata_df[metadata_df["well"] == well].copy()

    if len(well_metadata) == 0:
        print(f"No phenotype tiles found for well {well}")
        return {"total_translation": {}, "confidence": {well: {}}}

    coords = well_metadata[["x_pos", "y_pos"]].values
    tile_ids = well_metadata["tile"].values
    tile_size = (2400, 2400)  # Phenotype tile size in pixels

    print(
        f"Creating coordinate-based phenotype stitch config for {len(tile_ids)} tiles"
    )

    # Detect actual spacing between adjacent tiles
    distances = pdist(coords)
    actual_spacing = np.percentile(distances[distances > 0], 10)  # 10th percentile

    print(f"Phenotype detected spacing: {actual_spacing:.1f} μm")
    print(f"Phenotype tile size: {tile_size} pixels")

    # Determine pixel scaling from metadata or use fallback values
    if (
        "pixel_size_x" in well_metadata.columns
        and "pixel_size_y" in well_metadata.columns
    ):
        # Use pixel size from metadata (in μm per pixel)
        pixel_size_um = well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
        pixels_per_micron = 1.0 / pixel_size_um  # pixels per μm
        print(f"Phenotype pixel size from metadata: {pixel_size_um:.6f} μm/pixel")
        print(f"Phenotype pixels per micron: {pixels_per_micron:.4f}")

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
        # Fallback: Phenotype specs: 260 μm field of view, 2400 pixels -> 9.2308 pixels/μm
        phenotype_field_of_view_um = 260.0  # μm
        pixels_per_micron = tile_size[0] / phenotype_field_of_view_um
        print(f"Phenotype fallback - field of view: {phenotype_field_of_view_um} μm")
        print(f"Phenotype fallback - pixels per micron: {pixels_per_micron:.4f}")

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

    print(f"Generated {len(total_translation)} phenotype coordinate-based positions")

    # Verify output size and spacing
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Check for expected tile overlap
            expected_tile_spacing_pixels = actual_spacing * pixels_per_micron
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"Phenotype tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"Phenotype final image size: {final_size}")
    print(f"Phenotype memory estimate: {memory_gb:.1f} GB")

    return {"total_translation": total_translation, "confidence": {well: confidence}}


def _verify_stitch_quality(
    coords: np.ndarray,
    total_translation: Dict[str, list],
    tile_size: tuple,
    actual_spacing: float,
    pixels_per_micron: float,
    data_type: str
) -> None:
    """Verify the quality of stitching estimation by checking spacing and overlap.
    
    Args:
        coords: Array of stage coordinates
        total_translation: Dictionary of pixel translations
        tile_size: Tuple of (height, width) in pixels
        actual_spacing: Detected spacing between tiles in micrometers
        pixels_per_micron: Conversion factor from micrometers to pixels
        data_type: Type of data ("SBS" or "Phenotype") for logging
    """
    y_shifts = [shift[0] for shift in total_translation.values()]
    x_shifts = [shift[1] for shift in total_translation.values()]

    if len(y_shifts) > 1:
        pixel_spacings = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                stage_dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2
                    + (coords[i][1] - coords[j][1]) ** 2
                )
                pixel_dist = np.sqrt(
                    (y_shifts[i] - y_shifts[j]) ** 2 + (x_shifts[i] - x_shifts[j]) ** 2
                )
                if stage_dist > 0:
                    pixel_spacings.append(pixel_dist / stage_dist)

        if pixel_spacings:
            avg_pixel_spacing = np.mean(pixel_spacings)
            print(
                f"Verification - Average pixel spacing ratio: {avg_pixel_spacing:.4f} pixels/μm"
            )

            # Calculate actual average spacing between tiles
            actual_avg_spacing = np.mean(
                [
                    np.sqrt(
                        (y_shifts[i] - y_shifts[j]) ** 2
                        + (x_shifts[i] - x_shifts[j]) ** 2
                    )
                    for i in range(len(y_shifts))
                    for j in range(i + 1, len(y_shifts))
                ]
            )

            if actual_avg_spacing > 0:
                overlap_percent = (
                    (tile_size[0] - actual_avg_spacing) / tile_size[0] * 100
                )
                print(f"{data_type} tile overlap: {overlap_percent:.1f}%")
                if overlap_percent < 0:
                    print("⚠️  Warning: Negative overlap detected - tiles may have gaps")
                elif overlap_percent > 50:
                    print(
                        "⚠️  Warning: Very high overlap detected - may indicate scaling issues"
                    )

    final_size = (max(y_shifts) + tile_size[0], max(x_shifts) + tile_size[1])
    memory_gb = final_size[0] * final_size[1] * 2 / 1e9

    print(f"{data_type} final image size: {final_size}")
    print(f"{data_type} memory estimate: {memory_gb:.1f} GB")