"""Functions for coordinate scaling, triangle hashing, and alignment.

This module provides functionality for aligning cell positions between different
imaging modalities (e.g., phenotype and SBS) at the well level. The alignment process
includes coordinate scaling, triangle-based feature matching, and transformation
estimation using RANSAC.

Key components:
- High-level alignment pipeline via align_well_positions()
- Coordinate scaling based on field-of-view differences
- Triangle hashing for robust feature matching
- Adaptive regional sampling for improved alignment accuracy

Key parameters:
- score_threshold: Minimum alignment quality score to accept results (0.0-1.0)
- threshold_triangle: Maximum distance for triangle feature matching
- threshold_point: Maximum distance for point correspondence validation
"""

import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor

from lib.merge.hash import nine_edge_hash, get_vc, nearest_neighbors


def align_well_positions(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    score_threshold: float = 0.1,
    adaptive_region: bool = True,
    max_cells_for_hash: int = 75000,
    initial_region_size: int = 7000,
    min_triangles: int = 100,
    threshold_triangle: float = 0.3,
    threshold_point: float = 2.0,
) -> Dict[str, Any]:
    """Complete well-level alignment pipeline.

    Performs the full alignment workflow:
    1. Auto-calculate scale factor from coordinate ranges
    2. Scale phenotype coordinates to match SBS
    3. Generate triangle hash features
    4. Perform adaptive regional alignment
    5. Apply transformation

    Args:
        phenotype_positions: High-resolution phenotype cell positions with 'i', 'j' columns
        sbs_positions: Lower-resolution SBS cell positions with 'i', 'j' columns
        score_threshold: Minimum alignment score to accept (0.0-1.0)
        adaptive_region: Use adaptive regional sampling (recommended)
        max_cells_for_hash: Maximum cells for triangle generation
        initial_region_size: Starting region size for adaptive sampling
        min_triangles: Minimum triangles needed for reliable alignment
        threshold_triangle: Triangle similarity threshold
        threshold_point: Point distance threshold for validation

    Returns:
        Dictionary containing:
        - 'status': 'success', 'low_score', 'identity_fallback', or 'failed'
        - 'phenotype_scaled': Scaled phenotype coordinates
        - 'phenotype_transformed': Final transformed coordinates
        - 'phenotype_triangles': Triangle features for phenotype
        - 'sbs_triangles': Triangle features for SBS
        - 'alignment_params': Dict with rotation, translation, score, etc.
        - 'summary': Summary statistics for reporting
        - 'scale_factor': Calculated scale factor
        - 'overlap_fraction': Coordinate overlap fraction

    Raises:
        ValueError: If inputs have insufficient cells (< 4) for triangulation
    """
    print("=== WELL ALIGNMENT PIPELINE ===")
    print(f"Phenotype cells: {len(phenotype_positions):,}")
    print(f"SBS cells: {len(sbs_positions):,}")

    # Validate inputs
    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        raise ValueError(
            f"Insufficient cells for triangulation: "
            f"phenotype={len(phenotype_positions)}, sbs={len(sbs_positions)} (need ≥4 each)"
        )

    # Step 1: Calculate scale factor and scale coordinates
    print("\n--- Step 1: Coordinate Scaling ---")
    scale_factor = calculate_scale_factor_from_positions(
        phenotype_positions, sbs_positions
    )
    phenotype_scaled = scale_coordinates(phenotype_positions, scale_factor)
    overlap_fraction = calculate_coordinate_overlap(phenotype_scaled, sbs_positions)

    print(f"Scale factor: {scale_factor:.6f}")
    print(f"Coordinate overlap: {overlap_fraction:.1%} of SBS area")

    # Step 2: Generate triangle hashes
    print("\n--- Step 2: Triangle Hash Generation ---")
    phenotype_triangles = well_level_triangle_hash(phenotype_scaled)
    sbs_triangles = well_level_triangle_hash(sbs_positions)

    if len(phenotype_triangles) == 0 or len(sbs_triangles) == 0:
        print(" Triangle generation failed - using identity fallback")
        return _create_failed_result(
            phenotype_positions=phenotype_positions,
            phenotype_scaled=phenotype_scaled,
            sbs_positions=sbs_positions,
            scale_factor=scale_factor,
            overlap_fraction=overlap_fraction,
            failure_reason="insufficient_triangles",
        )

    print(f"Generated {len(phenotype_triangles):,} phenotype triangles")
    print(f"Generated {len(sbs_triangles):,} SBS triangles")

    # Step 3: Adaptive regional alignment
    print("\n--- Step 3: Triangle Hash Alignment ---")

    if not adaptive_region:
        print("Using adaptive regional sampling")

    alignment_result = _adaptive_regional_alignment(
        phenotype_scaled=phenotype_scaled,
        sbs_positions=sbs_positions,
        score_threshold=score_threshold,
        initial_region_size=initial_region_size,
        min_triangles=min_triangles,
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point,
    )

    # Step 4: Validate and finalize
    print("\n--- Step 4: Finalization ---")

    if alignment_result is None or alignment_result.empty:
        print(" Alignment failed - using identity fallback")
        return _create_failed_result(
            phenotype_positions=phenotype_positions,
            phenotype_scaled=phenotype_scaled,
            sbs_positions=sbs_positions,
            scale_factor=scale_factor,
            overlap_fraction=overlap_fraction,
            failure_reason="alignment_failed",
            phenotype_triangles=phenotype_triangles,
            sbs_triangles=sbs_triangles,
        )

    best_alignment = alignment_result.iloc[0]
    alignment_score = best_alignment.get("score", 0)

    # Determine status
    if alignment_score >= score_threshold:
        status = "success"
        print(f" Alignment successful: score={alignment_score:.3f}")
    else:
        status = "low_score"
        print(f"  Low alignment score: {alignment_score:.3f} < {score_threshold}")

    # Step 5: Apply transformation
    rotation = best_alignment.get("rotation", np.eye(2))
    translation = best_alignment.get("translation", np.array([0.0, 0.0]))

    if not isinstance(rotation, np.ndarray):
        rotation = np.eye(2)
    if not isinstance(translation, np.ndarray):
        translation = np.array([0.0, 0.0])

    phenotype_transformed = _apply_transformation(
        phenotype_scaled, rotation, translation
    )

    print(f"Applied transformation:")
    print(f"  Determinant: {np.linalg.det(rotation):.6f}")
    print(f"  Translation: [{translation[0]:.2f}, {translation[1]:.2f}]")

    # Prepare alignment parameters
    alignment_params = _prepare_alignment_params(
        rotation=rotation,
        translation=translation,
        score=alignment_score,
        scale_factor=scale_factor,
        best_alignment=best_alignment,
    )

    # Create summary for TSV output
    summary = _create_alignment_summary(
        status=status,
        scale_factor=scale_factor,
        overlap_fraction=overlap_fraction,
        phenotype_triangles=phenotype_triangles,
        sbs_triangles=sbs_triangles,
        rotation=rotation,
        translation=translation,
        alignment_score=alignment_score,
        score_threshold=score_threshold,
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point,
        best_alignment=best_alignment,
    )

    return {
        "status": status,
        "phenotype_scaled": phenotype_scaled,
        "phenotype_transformed": phenotype_transformed,
        "phenotype_triangles": phenotype_triangles,
        "sbs_triangles": sbs_triangles,
        "alignment_params": alignment_params,
        "summary": summary,
        "scale_factor": scale_factor,
        "overlap_fraction": overlap_fraction,
    }


def calculate_scale_factor_from_positions(
    phenotype_positions: pd.DataFrame, sbs_positions: pd.DataFrame
) -> float:
    """Calculate scale factor by comparing coordinate ranges from cell positions.

    The scale factor accounts for differences in magnification between imaging
    modalities. For example, 40x phenotype imaging compared to 10x SBS imaging
    would have an expected scale factor of approximately 0.25.

    Args:
        phenotype_positions: High-resolution phenotype cell positions with 'i', 'j' columns
        sbs_positions: Lower-resolution SBS cell positions with 'i', 'j' columns

    Returns:
        Scale factor to apply to phenotype coordinates (SBS_range / phenotype_range)

    Raises:
        ValueError: If coordinate ranges are zero or negative
    """
    # Get coordinate ranges for both datasets
    pheno_i_range = phenotype_positions["i"].max() - phenotype_positions["i"].min()
    pheno_j_range = phenotype_positions["j"].max() - phenotype_positions["j"].min()

    sbs_i_range = sbs_positions["i"].max() - sbs_positions["i"].min()
    sbs_j_range = sbs_positions["j"].max() - sbs_positions["j"].min()

    # Calculate scale factors for both dimensions
    scale_i = sbs_i_range / pheno_i_range if pheno_i_range > 0 else 0
    scale_j = sbs_j_range / pheno_j_range if pheno_j_range > 0 else 0

    print(f"Phenotype coordinate ranges: i={pheno_i_range:.0f}, j={pheno_j_range:.0f}")
    print(f"SBS coordinate ranges: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")
    print(f"Scale factors: i={scale_i:.6f}, j={scale_j:.6f}")

    # Use average for robustness (they should be approximately equal)
    scale_factor = (scale_i + scale_j) / 2

    return scale_factor


def scale_coordinates(positions_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    """Scale coordinate columns by a constant factor.

    Args:
        positions_df: DataFrame with 'i' and 'j' coordinate columns
        scale_factor: Factor to multiply coordinates by

    Returns:
        DataFrame with scaled coordinates (copy of original with modified coordinates)
    """
    scaled_df = positions_df.copy()
    scaled_df["i"] = scaled_df["i"] * scale_factor
    scaled_df["j"] = scaled_df["j"] * scale_factor
    return scaled_df


def calculate_coordinate_overlap(
    positions_1: pd.DataFrame, positions_2: pd.DataFrame
) -> float:
    """Calculate the overlap fraction between two coordinate datasets.

    Computes the rectangular overlap between the bounding boxes of two
    coordinate datasets as a fraction of the second dataset's area.

    Args:
        positions_1: First coordinate dataset with 'i', 'j' columns
        positions_2: Second coordinate dataset with 'i', 'j' columns

    Returns:
        Overlap fraction (0.0 to 1.0), where 1.0 means complete overlap
        of the second dataset within the first
    """
    # Get coordinate ranges
    i1_min, i1_max = positions_1["i"].min(), positions_1["i"].max()
    j1_min, j1_max = positions_1["j"].min(), positions_1["j"].max()

    i2_min, i2_max = positions_2["i"].min(), positions_2["i"].max()
    j2_min, j2_max = positions_2["j"].min(), positions_2["j"].max()

    # Calculate overlap region
    overlap_i_min = max(i1_min, i2_min)
    overlap_i_max = min(i1_max, i2_max)
    overlap_j_min = max(j1_min, j2_min)
    overlap_j_max = min(j1_max, j2_max)

    # Check if there's any overlap
    if overlap_i_max <= overlap_i_min or overlap_j_max <= overlap_j_min:
        return 0.0

    # Calculate overlap area
    overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)

    # Calculate reference area (use second dataset)
    ref_area = (i2_max - i2_min) * (j2_max - j2_min)

    if ref_area <= 0:
        return 0.0

    return overlap_area / ref_area


def well_level_triangle_hash(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Generate triangle hash for well-level data using Delaunay triangulation.

    Creates a robust feature representation by computing triangle edge vectors
    from Delaunay triangulation. Each triangle is represented by a 18-dimensional
    vector of edge lengths and orientations, making it suitable for matching
    across different coordinate systems.

    Args:
        positions_df: DataFrame with 'i', 'j' coordinate columns

    Returns:
        DataFrame with triangle feature vectors including:
        - V_0 to V_17: Triangle edge vector components
        - c_0, c_1: Triangle centroid coordinates
        - magnitude: Triangle scale normalization factor

        Returns empty DataFrame if triangulation fails or insufficient points
    """
    if len(positions_df) < 4:
        return pd.DataFrame()

    coords = positions_df[["i", "j"]].values

    try:
        dt = Delaunay(coords)
        vectors, centers = [], []

        for i in range(dt.simplices.shape[0]):
            # Skip triangles with an edge on the outer boundary
            if (dt.neighbors[i] == -1).any():
                continue

            # Generate nine-edge hash feature vector
            result = nine_edge_hash(dt, i)
            if result is None:
                continue

            _, v = result
            c = coords[dt.simplices[i], :].mean(axis=0)
            vectors.append(v)
            centers.append(c)

        if not vectors:
            return pd.DataFrame()

        # Convert to standardized format
        vectors_array = np.array(vectors).reshape(-1, 18)
        centers_array = np.array(centers)

        # Create DataFrame with consistent column naming
        df_vectors = pd.DataFrame(vectors_array).rename(columns=lambda x: f"V_{x}")
        df_coords = pd.DataFrame(centers_array).rename(columns=lambda x: f"c_{x}")
        df_combined = pd.concat([df_vectors, df_coords], axis=1)

        # Add magnitude column for normalization
        df_result = df_combined.assign(
            magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5")
        )

        return df_result

    except Exception as e:
        print(f"Triangle hash generation failed: {e}")
        return pd.DataFrame()


def evaluate_well_match(
    vec_centers_0: pd.DataFrame,
    vec_centers_1: pd.DataFrame,
    threshold_triangle: float = 0.3,
    threshold_point: float = 2.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Evaluate alignment match using triangle hash features and RANSAC.

    Uses a single fixed random seed for reproducible, fast alignment evaluation.
    Matches triangle features between datasets and estimates the best rigid
    transformation using RANSAC regression.

    Args:
        vec_centers_0: First dataset triangle features
        vec_centers_1: Second dataset triangle features
        threshold_triangle: Maximum distance for triangle feature matching
        threshold_point: Maximum distance for point correspondence validation

    Returns:
        Tuple of (rotation_matrix, translation_vector, alignment_score):
        - rotation_matrix: 2x2 rotation/scaling matrix, None if failed
        - translation_vector: 2D translation vector, None if failed
        - alignment_score: Quality score (0-1), -1 if failed
    """
    V_0, c_0 = get_vc(vec_centers_0)
    V_1, c_1 = get_vc(vec_centers_1)
    i0, i1, distances = nearest_neighbors(V_0, V_1)

    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]

    if sum(filt) < 5:
        return None, None, -1

    # Use fixed seed for reproducible results
    print(f"Using single RANSAC seed (42) for fast evaluation...")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(
                random_state=42,
                min_samples=max(5, len(X) // 10),
                max_trials=1000,
            )
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_
        determinant = np.linalg.det(rotation)

        # Calculate alignment quality score
        distances = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances.min(axis=0)) < threshold_region
        score = (np.sqrt(distances.min(axis=0))[filt_score] < threshold_point).mean()

        print(f" Single seed (42) result: det={determinant:.6f}, score={score:.3f}")

        return rotation, translation, score

    except Exception as e:
        print(f" RANSAC with seed 42 failed: {e}")
        return None, None, -1


def _adaptive_regional_alignment(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    score_threshold: float,
    initial_region_size: int,
    min_triangles: int,
    threshold_triangle: float,
    threshold_point: float,
) -> Optional[pd.DataFrame]:
    """Perform adaptive regional triangle hash alignment.

    Iteratively increases region size until sufficient triangles are found
    for reliable alignment.

    Returns:
        DataFrame with alignment results or None if failed
    """
    region_size = initial_region_size
    max_region_size = (
        min(
            phenotype_scaled["i"].max() - phenotype_scaled["i"].min(),
            phenotype_scaled["j"].max() - phenotype_scaled["j"].min(),
            sbs_positions["i"].max() - sbs_positions["i"].min(),
            sbs_positions["j"].max() - sbs_positions["j"].min(),
        )
        * 0.8
    )

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts and region_size <= max_region_size:
        attempts += 1
        print(f"\n--- Attempt {attempts}: Region size {region_size:.0f} ---")

        # Sample region
        pheno_region, sbs_region, region_info = _sample_region_for_alignment(
            phenotype_scaled, sbs_positions, region_size=int(region_size)
        )

        if len(pheno_region) < 4 or len(sbs_region) < 4:
            print(
                f"Insufficient cells in region ({len(pheno_region)}, {len(sbs_region)}), increasing region size"
            )
            region_size *= 1.5
            continue

        # Generate triangle hashes for the region
        pheno_triangles = well_level_triangle_hash(pheno_region)
        sbs_triangles = well_level_triangle_hash(sbs_region)

        if len(pheno_triangles) < min_triangles or len(sbs_triangles) < min_triangles:
            print(
                f"Insufficient triangles ({len(pheno_triangles)}, {len(sbs_triangles)}), increasing region size"
            )
            region_size *= 1.5
            continue

        print(
            f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles"
        )

        # Evaluate triangle hash match
        rotation, translation, calculated_score = evaluate_well_match(
            pheno_triangles,
            sbs_triangles,
            threshold_triangle=threshold_triangle,
            threshold_point=threshold_point,
        )

        if rotation is None or calculated_score < score_threshold:
            print(
                f"Triangle hash match failed: score={calculated_score:.3f} < {score_threshold}, increasing region size"
            )
            region_size *= 1.5
            continue

        # Success!
        determinant = np.linalg.det(rotation)

        print(f"✅ Regional triangle hash alignment successful:")
        print(f"   Score: {calculated_score:.3f} (threshold: {score_threshold})")
        print(f"   Determinant: {determinant:.6f}")
        print(f"   Region size used: {region_size:.0f}")

        # Build result
        alignment = {
            "rotation": rotation,
            "translation": translation,
            "score": calculated_score,
            "determinant": determinant,
            "transformation_type": "triangle_hash_regional",
            "triangles_matched": min(len(pheno_triangles), len(sbs_triangles)),
            "approach": "adaptive_regional_sampling",
            "region_info": region_info,
            "attempts": attempts,
            "final_region_size": region_size,
        }

        return pd.DataFrame([alignment])

    # All attempts failed
    print(f"❌ Regional triangle hash failed after {attempts} attempts")
    print(f"Final region size tried: {region_size:.0f}")

    return None


def _sample_region_for_alignment(
    phenotype_resized: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    region_size: int = 7000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Sample a centered region from both datasets for triangle hash alignment.

    Creates a square sampling region centered on the overlap between datasets.

    Args:
        phenotype_resized: Resized phenotype positions with 'i', 'j' columns
        sbs_positions: SBS positions with 'i', 'j' columns
        region_size: Size of the square region to sample (in coordinate units)

    Returns:
        Tuple of (phenotype_region, sbs_region, region_info)
    """
    print(f"Regional sampling: {region_size}x{region_size}")

    # Find overlap bounds
    pheno_i_min, pheno_i_max = (
        phenotype_resized["i"].min(),
        phenotype_resized["i"].max(),
    )
    pheno_j_min, pheno_j_max = (
        phenotype_resized["j"].min(),
        phenotype_resized["j"].max(),
    )

    sbs_i_min, sbs_i_max = sbs_positions["i"].min(), sbs_positions["i"].max()
    sbs_j_min, sbs_j_max = sbs_positions["j"].min(), sbs_positions["j"].max()

    # Calculate overlap region
    overlap_i_min = max(pheno_i_min, sbs_i_min)
    overlap_i_max = min(pheno_i_max, sbs_i_max)
    overlap_j_min = max(pheno_j_min, sbs_j_min)
    overlap_j_max = min(pheno_j_max, sbs_j_max)

    if overlap_i_max <= overlap_i_min or overlap_j_max <= overlap_j_min:
        print("❌ No overlap found between datasets")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Center the sampling region
    center_i = (overlap_i_min + overlap_i_max) / 2
    center_j = (overlap_j_min + overlap_j_max) / 2
    half_size = region_size / 2

    # Define sampling bounds
    sample_i_min = max(overlap_i_min, center_i - half_size)
    sample_i_max = min(overlap_i_max, center_i + half_size)
    sample_j_min = max(overlap_j_min, center_j - half_size)
    sample_j_max = min(overlap_j_max, center_j + half_size)

    # Sample cells
    pheno_mask = (
        (phenotype_resized["i"] >= sample_i_min)
        & (phenotype_resized["i"] <= sample_i_max)
        & (phenotype_resized["j"] >= sample_j_min)
        & (phenotype_resized["j"] <= sample_j_max)
    )

    sbs_mask = (
        (sbs_positions["i"] >= sample_i_min)
        & (sbs_positions["i"] <= sample_i_max)
        & (sbs_positions["j"] >= sample_j_min)
        & (sbs_positions["j"] <= sample_j_max)
    )

    pheno_region = phenotype_resized[pheno_mask].copy()
    sbs_region = sbs_positions[sbs_mask].copy()

    print(f"Sampled: {len(pheno_region):,} phenotype, {len(sbs_region):,} SBS")

    region_info = {
        "region_size": region_size,
        "bounds": {
            "i_min": sample_i_min,
            "i_max": sample_i_max,
            "j_min": sample_j_min,
            "j_max": sample_j_max,
        },
        "cell_counts": {"phenotype": len(pheno_region), "sbs": len(sbs_region)},
    }

    return pheno_region, sbs_region, region_info


def _apply_transformation(
    positions: pd.DataFrame,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> pd.DataFrame:
    """Apply rotation and translation to coordinates.

    Args:
        positions: DataFrame with 'i', 'j' columns
        rotation: 2x2 rotation matrix
        translation: 2D translation vector

    Returns:
        DataFrame with transformed coordinates
    """
    transformed = positions.copy()
    coords = positions[["i", "j"]].values
    transformed_coords = coords @ rotation.T + translation
    transformed["i"] = transformed_coords[:, 0]
    transformed["j"] = transformed_coords[:, 1]
    return transformed


def _prepare_alignment_params(
    rotation: np.ndarray,
    translation: np.ndarray,
    score: float,
    scale_factor: float,
    best_alignment: pd.Series,
) -> pd.DataFrame:
    """Prepare alignment parameters for Parquet storage.

    Returns:
        DataFrame with alignment parameters
    """

    def safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    rotation_flat = rotation.flatten().astype(float).tolist()
    translation_list = translation.astype(float).tolist()

    params = {
        "rotation_matrix_flat": rotation_flat,
        "translation_vector": translation_list,
        "scale_factor": safe_float(scale_factor),
        "score": safe_float(score),
        "determinant": safe_float(np.linalg.det(rotation)),
        "transformation_type": str(
            best_alignment.get("transformation_type", "unknown")
        ),
        "approach": str(best_alignment.get("approach", "unknown")),
        "validation_mean_distance": safe_float(
            best_alignment.get("validation_mean_distance", 0.0)
        ),
        "validation_median_distance": safe_float(
            best_alignment.get("validation_median_distance", 0.0)
        ),
        "has_overlap": bool(best_alignment.get("has_overlap", True)),
    }

    return pd.DataFrame([params])


def _create_alignment_summary(
    status: str,
    scale_factor: float,
    overlap_fraction: float,
    phenotype_triangles: pd.DataFrame,
    sbs_triangles: pd.DataFrame,
    rotation: np.ndarray,
    translation: np.ndarray,
    alignment_score: float,
    score_threshold: float,
    threshold_triangle: float,
    threshold_point: float,
    best_alignment: pd.Series,
) -> Dict[str, Any]:
    """Create summary dictionary for TSV output.

    Returns:
        Dictionary with summary statistics
    """

    def safe_float(value, default: float = 0.0, precision: int = 6) -> float:
        try:
            if value is None:
                return default
            return round(float(value), precision)
        except (ValueError, TypeError):
            return default

    def safe_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    # Extract rotation components
    r00, r01 = float(rotation[0, 0]), float(rotation[0, 1])
    r10, r11 = float(rotation[1, 0]), float(rotation[1, 1])

    # Extract translation components
    tx, ty = float(translation[0]), float(translation[1])

    return {
        "status": status,
        "failure_reason": "",
        "scale_factor": safe_float(scale_factor),
        "overlap_fraction": safe_float(overlap_fraction, precision=3),
        "phenotype_triangles": safe_int(len(phenotype_triangles)),
        "sbs_triangles": safe_int(len(sbs_triangles)),
        "threshold_triangle": threshold_triangle,
        "score_threshold": safe_float(score_threshold, precision=3),
        "threshold_point": threshold_point,
        "approach": str(best_alignment.get("approach", "")),
        "transformation_type": str(best_alignment.get("transformation_type", "")),
        "alignment_score": safe_float(alignment_score, precision=3),
        "determinant": safe_float(np.linalg.det(rotation), precision=6),
        "rotation_r00": safe_float(r00, precision=6),
        "rotation_r01": safe_float(r01, precision=6),
        "rotation_r10": safe_float(r10, precision=6),
        "rotation_r11": safe_float(r11, precision=6),
        "translation_tx": safe_float(tx, precision=3),
        "translation_ty": safe_float(ty, precision=3),
        "validation_mean_distance": safe_float(
            best_alignment.get("validation_mean_distance", 0), precision=3
        ),
        "validation_median_distance": safe_float(
            best_alignment.get("validation_median_distance", 0), precision=3
        ),
        "region_size": safe_float(
            best_alignment.get("final_region_size", 0), precision=0
        ),
        "sampling_attempts": safe_int(best_alignment.get("attempts", 0)),
        "triangles_matched": safe_int(best_alignment.get("triangles_matched", 0)),
    }


def _create_failed_result(
    phenotype_positions: pd.DataFrame,
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float,
    overlap_fraction: float,
    failure_reason: str,
    phenotype_triangles: Optional[pd.DataFrame] = None,
    sbs_triangles: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Create a failed alignment result with identity transformation.

    Args:
        phenotype_positions: Original phenotype positions
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        scale_factor: Calculated scale factor
        overlap_fraction: Coordinate overlap fraction
        failure_reason: Reason for failure
        phenotype_triangles: Triangle features if available
        sbs_triangles: Triangle features if available

    Returns:
        Dictionary with failed alignment result
    """
    # Use identity transformation
    rotation = np.eye(2)
    translation = np.array([0.0, 0.0])

    # Validate with a sample to estimate alignment quality
    sample_size = min(1000, len(phenotype_scaled), len(sbs_positions))
    if sample_size > 0:
        pheno_sample = phenotype_scaled.sample(n=sample_size)[["i", "j"]].values
        sbs_sample = sbs_positions.sample(n=sample_size)[["i", "j"]].values
        distances = cdist(pheno_sample, sbs_sample, metric="euclidean")
        min_distances = distances.min(axis=1)
        score_val = (min_distances < 10.0).mean()
        mean_dist = min_distances.mean()
        median_dist = np.median(min_distances)
    else:
        score_val = 0.0
        mean_dist = 0.0
        median_dist = 0.0

    # Create empty triangles if not provided
    if phenotype_triangles is None:
        phenotype_triangles = pd.DataFrame(
            columns=["V_0", "V_1", "c_0", "c_1", "magnitude"]
        )
    if sbs_triangles is None:
        sbs_triangles = pd.DataFrame(columns=["V_0", "V_1", "c_0", "c_1", "magnitude"])

    # Alignment parameters
    alignment_params = pd.DataFrame(
        [
            {
                "rotation_matrix_flat": [1.0, 0.0, 0.0, 1.0],
                "translation_vector": [0.0, 0.0],
                "score": score_val,
                "determinant": 1.0,
                "transformation_type": f"failed_{failure_reason}",
                "scale_factor": float(scale_factor),
                "approach": "identity_fallback",
                "validation_mean_distance": mean_dist,
                "validation_median_distance": median_dist,
                "has_overlap": overlap_fraction > 0,
            }
        ]
    )

    # Summary
    summary = {
        "status": "failed",
        "failure_reason": failure_reason,
        "scale_factor": float(scale_factor),
        "overlap_fraction": round(overlap_fraction, 3),
        "phenotype_triangles": len(phenotype_triangles),
        "sbs_triangles": len(sbs_triangles),
        "threshold_triangle": 0.3,
        "score_threshold": 0.1,
        "threshold_point": 2.0,
        "approach": "identity_fallback",
        "transformation_type": f"failed_{failure_reason}",
        "alignment_score": score_val,
        "determinant": 1.0,
        "rotation_r00": 1.0,
        "rotation_r01": 0.0,
        "rotation_r10": 0.0,
        "rotation_r11": 1.0,
        "translation_tx": 0.0,
        "translation_ty": 0.0,
        "validation_mean_distance": mean_dist,
        "validation_median_distance": median_dist,
        "region_size": 0.0,
        "sampling_attempts": 0,
        "triangles_matched": 0,
    }

    # Use scaled coordinates as "transformed" (no additional transformation)
    phenotype_transformed = phenotype_scaled.copy()

    return {
        "status": "failed",
        "phenotype_scaled": phenotype_scaled,
        "phenotype_transformed": phenotype_transformed,
        "phenotype_triangles": phenotype_triangles,
        "sbs_triangles": sbs_triangles,
        "alignment_params": alignment_params,
        "summary": summary,
        "scale_factor": scale_factor,
        "overlap_fraction": overlap_fraction,
    }
