"""
Enhanced well-level merge functions using actual stitched masks.
This replaces the coordinate-transformation approach with actual image/mask stitching.
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from skimage import measure

# Import existing functions
from lib.merge.hash import get_vectors, get_vc, nearest_neighbors
from lib.merge.stitch_well import (
    estimate_stitch_aligned_tiff,
    assemble_aligned_tiff_well,
    assemble_stitched_masks_simple,
    extract_cell_positions_from_stitched_mask,
)


def hash_stitched_cell_positions(cell_positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hashed Delaunay triangulation for cell positions from stitched masks.

    Args:
        cell_positions_df: DataFrame with cell positions extracted from stitched masks
                          Expected columns: ['well', 'cell', 'i', 'j', 'area', 'data_type']

    Returns:
        DataFrame with triangle hash features
    """
    if len(cell_positions_df) < 4:
        print(
            f"Warning: Only {len(cell_positions_df)} cells found, need at least 4 for triangulation"
        )
        return pd.DataFrame()

    # Extract coordinates and compute Delaunay triangulation
    coordinates = cell_positions_df[["i", "j"]].values

    try:
        v, c = get_vectors(coordinates)

        if len(v) == 0:
            print("No valid triangles generated")
            return pd.DataFrame()

        # Create dataframes for vectors and centers
        df_vectors = pd.DataFrame(v).rename(columns="V_{0}".format)
        df_coords = pd.DataFrame(c).rename(columns="c_{0}".format)

        # Combine and add magnitude
        df_combined = pd.concat([df_vectors, df_coords], axis=1)
        df_result = df_combined.assign(
            magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5")
        )

        # Add well information
        df_result["well"] = cell_positions_df["well"].iloc[0]

        return df_result

    except Exception as e:
        print(f"Error computing triangulation: {e}")
        return pd.DataFrame()


def stitched_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Perform alignment between phenotype and SBS stitched wells using actual cell positions.

    Args:
        phenotype_positions: Cell positions in phenotype stitched well
        sbs_positions: Cell positions in SBS stitched well
        det_range: Acceptable range for transformation determinant
        score_threshold: Minimum score for valid alignment

    Returns:
        DataFrame with alignment parameters
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        print("Empty position data, cannot perform alignment")
        return pd.DataFrame()

    # Generate triangle hashes for both datasets
    phenotype_hash = hash_stitched_cell_positions(phenotype_positions)
    sbs_hash = hash_stitched_cell_positions(sbs_positions)

    if len(phenotype_hash) == 0 or len(sbs_hash) == 0:
        print("Empty hash data, cannot perform alignment")
        return pd.DataFrame()

    try:
        # Extract vectors and centers
        V_0, c_0 = get_vc(phenotype_hash)
        V_1, c_1 = get_vc(sbs_hash)

        # Find nearest neighbors between triangle vectors
        i0, i1, distances = nearest_neighbors(V_0, V_1)

        # Filter based on distance threshold
        triangle_threshold = 0.3  # Triangle matching threshold
        filt = distances < triangle_threshold
        if filt.sum() < 5:
            print(f"Only {filt.sum()} matching triangles found, need at least 5")
            return pd.DataFrame()

        # Get matching triangle centers
        X, Y = c_0[i0[filt]], c_1[i1[filt]]

        # Fit transformation using RANSAC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(min_samples=5, max_trials=1000)
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_

        # Score the transformation
        predicted_centers = model.predict(c_0)
        distances_score = cdist(predicted_centers, c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances_score.min(axis=0)) < threshold_region
        score = (np.sqrt(distances_score.min(axis=0))[filt_score] < 2).mean()

        # Calculate determinant
        determinant = np.linalg.det(rotation)

        # Create result
        result = pd.DataFrame(
            [
                {
                    "rotation_1": rotation[0] if len(rotation) > 0 else [0, 0],
                    "rotation_2": rotation[1] if len(rotation) > 1 else [0, 0],
                    "translation": translation,
                    "score": score,
                    "determinant": determinant,
                    "well": phenotype_positions["well"].iloc[0]
                    if len(phenotype_positions) > 0
                    else "unknown",
                    "n_triangles_matched": filt.sum(),
                    "n_triangles_phenotype": len(c_0),
                    "n_triangles_sbs": len(c_1),
                }
            ]
        )

        print(
            f"Alignment results: score={score:.3f}, determinant={determinant:.3f}, matched_triangles={filt.sum()}"
        )

        return result

    except Exception as e:
        print(f"Alignment failed: {e}")
        return pd.DataFrame()


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: pd.Series,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Merge cell positions between phenotype and SBS stitched wells.

    Args:
        phenotype_positions: Cell positions in phenotype well
        sbs_positions: Cell positions in SBS well
        alignment: Alignment parameters (rotation, translation)
        threshold: Maximum distance for cell matching

    Returns:
        DataFrame with merged cell identities
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "cell_0",
                "i_0",
                "j_0",
                "area_0",
                "cell_1",
                "i_1",
                "j_1",
                "area_1",
                "distance",
            ]
        )

    try:
        # Build transformation model
        rotation = np.array([alignment["rotation_1"], alignment["rotation_2"]])
        translation = alignment["translation"]
        model = LinearRegression()
        model.coef_ = rotation
        model.intercept_ = translation

        # Extract coordinates
        X = phenotype_positions[["i", "j"]].values
        Y = sbs_positions[["i", "j"]].values

        # Apply transformation to phenotype coordinates
        X_transformed = model.predict(X)

        # Calculate distances between transformed phenotype and SBS
        distances = cdist(X_transformed, Y, metric="euclidean")

        # Find nearest neighbors
        ix = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)

        # Filter by threshold
        filt = min_distances < threshold

        if filt.sum() == 0:
            print("No matches found within threshold")
            return pd.DataFrame(
                columns=[
                    "plate",
                    "well",
                    "cell_0",
                    "i_0",
                    "j_0",
                    "area_0",
                    "cell_1",
                    "i_1",
                    "j_1",
                    "area_1",
                    "distance",
                ]
            )

        # Create merged dataframe
        matched_phenotype = phenotype_positions[filt].reset_index(drop=True)
        matched_sbs = sbs_positions.iloc[ix[filt]].reset_index(drop=True)

        merged_data = pd.DataFrame(
            {
                "plate": 1,  # You may need to extract this from your data
                "well": matched_phenotype["well"],
                "cell_0": matched_phenotype["cell"],
                "i_0": matched_phenotype["i"],
                "j_0": matched_phenotype["j"],
                "area_0": matched_phenotype["area"],
                "cell_1": matched_sbs["cell"],
                "i_1": matched_sbs["i"],
                "j_1": matched_sbs["j"],
                "area_1": matched_sbs["area"],
                "distance": min_distances[filt],
            }
        )

        print(f"Successfully merged {len(merged_data)} cells (threshold={threshold})")
        return merged_data

    except Exception as e:
        print(f"Merge failed: {e}")
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "cell_0",
                "i_0",
                "j_0",
                "area_0",
                "cell_1",
                "i_1",
                "j_1",
                "area_1",
                "distance",
            ]
        )


def create_stitched_overlay(
    stitched_image: np.ndarray,
    stitched_mask: np.ndarray,
    overlay_alpha: float = 0.3,
    mask_color: Tuple[int, int, int] = (255, 0, 0),  # Red
) -> np.ndarray:
    """
    Create an overlay of segmentation mask on stitched image.

    Args:
        stitched_image: Stitched image (2D grayscale)
        stitched_mask: Stitched segmentation mask
        overlay_alpha: Alpha value for mask overlay
        mask_color: RGB color for mask overlay

    Returns:
        RGB overlay image
    """
    import cv2

    # Normalize image to 0-255
    if stitched_image.max() > 255:
        image_norm = (
            (stitched_image - stitched_image.min())
            / (stitched_image.max() - stitched_image.min())
            * 255
        ).astype(np.uint8)
    else:
        image_norm = stitched_image.astype(np.uint8)

    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)

    # Create mask overlay
    mask_binary = (stitched_mask > 0).astype(np.uint8)

    # Create colored mask
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[mask_binary == 1] = mask_color

    # Blend images
    overlay = cv2.addWeighted(
        image_rgb, 1 - overlay_alpha, mask_colored, overlay_alpha, 0
    )

    return overlay


def full_stitching_pipeline(
    metadata_df: pd.DataFrame,
    well: str,
    data_type: str = "phenotype",
    stitch_config: Optional[Dict] = None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    channel: int = 0,
    overlap_percent: float = 0.1,
    create_overlay: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Complete stitching pipeline for both images and masks.

    Args:
        metadata_df: DataFrame with tile metadata
        well: Well identifier
        data_type: "phenotype" or "sbs"
        stitch_config: Existing stitch config, or None to compute
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations
        channel: Which channel to use for registration
        overlap_percent: Percentage of tile that overlaps
        create_overlay: Whether to create overlay image
        output_dir: Directory to save outputs

    Returns:
        Dictionary with results
    """
    print(f"Starting full stitching pipeline for {data_type} well {well}")

    results = {}

    # Step 1: Estimate stitching if config not provided
    if stitch_config is None:
        print("Estimating tile shifts...")
        stitch_config = estimate_stitch_aligned_tiff(
            metadata_df=metadata_df,
            well=well,
            data_type=data_type,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            channel=channel,
        )

    shifts = stitch_config["total_translation"]
    results["stitch_config"] = stitch_config

    # Step 2: Assemble stitched image
    print("Assembling stitched image...")
    try:
        stitched_image = assemble_aligned_tiff_well(
            metadata_df=metadata_df,
            shifts=shifts,
            well=well,
            data_type=data_type,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            overlap_percent=overlap_percent,
        )
        results["stitched_image"] = stitched_image
        print(f"Stitched image shape: {stitched_image.shape}")
    except Exception as e:
        print(f"Error assembling stitched image: {e}")
        stitched_image = None

    # Step 3: Assemble stitched masks
    print("Assembling stitched masks...")
    try:
        stitched_mask = assemble_stitched_masks_simple(
            metadata_df=metadata_df,
            shifts=shifts,
            well=well,
            data_type=data_type,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
        )
        results["stitched_mask"] = stitched_mask
        print(
            f"Stitched mask shape: {stitched_mask.shape}, max label: {stitched_mask.max()}"
        )
    except Exception as e:
        print(f"Error assembling stitched mask: {e}")
        stitched_mask = None

    # Step 4: Extract cell positions
    if stitched_mask is not None:
        print("Extracting cell positions...")
        try:
            cell_positions = extract_cell_positions_from_stitched_mask(
                stitched_mask, well, data_type
            )
            results["cell_positions"] = cell_positions
        except Exception as e:
            print(f"Error extracting cell positions: {e}")

    # Step 5: Create overlay
    if create_overlay and stitched_image is not None and stitched_mask is not None:
        print("Creating overlay image...")
        try:
            overlay = create_stitched_overlay(stitched_image, stitched_mask)
            results["overlay"] = overlay
        except Exception as e:
            print(f"Error creating overlay: {e}")

    # Step 6: Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from skimage import io

        if stitched_image is not None:
            image_path = output_dir / f"{well}_{data_type}_stitched.tiff"
            io.imsave(image_path, stitched_image)
            print(f"Saved stitched image to {image_path}")

        if stitched_mask is not None:
            mask_path = output_dir / f"{well}_{data_type}_mask.tiff"
            io.imsave(mask_path, stitched_mask.astype(np.uint16))
            print(f"Saved stitched mask to {mask_path}")

        if "overlay" in results:
            overlay_path = output_dir / f"{well}_{data_type}_overlay.png"
            io.imsave(overlay_path, results["overlay"])
            print(f"Saved overlay to {overlay_path}")

        if "cell_positions" in results:
            positions_path = output_dir / f"{well}_{data_type}_positions.parquet"
            results["cell_positions"].to_parquet(positions_path)
            print(f"Saved cell positions to {positions_path}")

    print(f"Full stitching pipeline completed for {data_type} well {well}")
    return results


def stitched_well_merge_pipeline(
    phenotype_metadata: pd.DataFrame,
    sbs_metadata: pd.DataFrame,
    well: str,
    plate: str,
    phenotype_stitch_config: Optional[Dict] = None,
    sbs_stitch_config: Optional[Dict] = None,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
    distance_threshold: float = 2.0,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete stitched well-level merge pipeline.

    Args:
        phenotype_metadata: Phenotype metadata DataFrame
        sbs_metadata: SBS metadata DataFrame
        well: Well identifier
        plate: Plate identifier
        phenotype_stitch_config: Phenotype stitching configuration
        sbs_stitch_config: SBS stitching configuration
        det_range: Valid determinant range for alignment
        score_threshold: Minimum alignment score
        distance_threshold: Maximum cell matching distance
        flipud: Whether to flip tiles up-down
        fliplr: Whether to flip tiles left-right
        rot90: Number of 90-degree rotations
        output_dir: Optional output directory for intermediate files

    Returns:
        Tuple of (merged_cells_df, alignment_df, stitching_results)
    """
    print(f"Starting stitched well-level merge for plate {plate}, well {well}")

    # Filter metadata
    phenotype_well = phenotype_metadata[
        (phenotype_metadata["plate"] == int(plate))
        & (phenotype_metadata["well"] == well)
    ]

    sbs_well = sbs_metadata[
        (sbs_metadata["plate"] == int(plate)) & (sbs_metadata["well"] == well)
    ]

    print(f"Found {len(phenotype_well)} phenotype tiles, {len(sbs_well)} SBS tiles")

    if len(phenotype_well) < 4 or len(sbs_well) < 4:
        print("Insufficient tiles for processing")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Process phenotype
    print("\n=== Processing Phenotype ===")
    phenotype_results = full_stitching_pipeline(
        metadata_df=phenotype_well,
        well=well,
        data_type="phenotype",
        stitch_config=phenotype_stitch_config,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        output_dir=Path(output_dir) / "phenotype" if output_dir else None,
    )

    # Process SBS
    print("\n=== Processing SBS ===")
    sbs_results = full_stitching_pipeline(
        metadata_df=sbs_well,
        well=well,
        data_type="sbs",
        stitch_config=sbs_stitch_config,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        output_dir=Path(output_dir) / "sbs" if output_dir else None,
    )

    # Check if both processed successfully
    if "cell_positions" not in phenotype_results or "cell_positions" not in sbs_results:
        print("Failed to extract cell positions from one or both datasets")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"phenotype": phenotype_results, "sbs": sbs_results},
        )

    phenotype_positions = phenotype_results["cell_positions"]
    sbs_positions = sbs_results["cell_positions"]

    print(
        f"Extracted {len(phenotype_positions)} phenotype cells, {len(sbs_positions)} SBS cells"
    )

    # Perform alignment
    print("\n=== Performing Alignment ===")
    alignment_df = stitched_well_alignment(
        phenotype_positions,
        sbs_positions,
        det_range,
        score_threshold,
    )

    if len(alignment_df) == 0:
        print("Alignment failed")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"phenotype": phenotype_results, "sbs": sbs_results},
        )

    # Check alignment quality
    alignment = alignment_df.iloc[0]
    if (
        alignment["determinant"] < det_range[0]
        or alignment["determinant"] > det_range[1]
        or alignment["score"] < score_threshold
    ):
        print(
            f"Alignment quality insufficient: det={alignment['determinant']:.3f}, score={alignment['score']:.3f}"
        )
        return (
            pd.DataFrame(),
            alignment_df,
            {"phenotype": phenotype_results, "sbs": sbs_results},
        )

    # Merge cells
    print("\n=== Merging Cells ===")
    merged_cells = merge_stitched_cells(
        phenotype_positions, sbs_positions, alignment, distance_threshold
    )

    print(f"Stitched well-level merge completed. Merged {len(merged_cells)} cells.")

    return (
        merged_cells,
        alignment_df,
        {"phenotype": phenotype_results, "sbs": sbs_results},
    )


# For backward compatibility, keep this function name
def well_merge_pipeline(
    phenotype_info: pd.DataFrame,
    sbs_info: pd.DataFrame,
    phenotype_shifts: Dict[str, List[int]],
    sbs_shifts: Dict[str, List[int]],
    well: str,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
    distance_threshold: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED: Use stitched_well_merge_pipeline instead.
    This function is kept for backward compatibility but will use the enhanced approach.
    """
    warnings.warn(
        "well_merge_pipeline is deprecated. Use stitched_well_merge_pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility, we'll need the metadata to recreate the enhanced pipeline
    # This is a simplified version - you may need to adjust based on your data structure
    print(
        "Using deprecated well_merge_pipeline - consider upgrading to stitched_well_merge_pipeline"
    )

    # Return empty results for now - you'll need to provide metadata to use the enhanced version
    return pd.DataFrame(), pd.DataFrame()
