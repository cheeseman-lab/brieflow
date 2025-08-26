"""Step 1 Library: Coordinate scaling, triangle hashing, and alignment functions.
Save this as: lib/merge/alignment.py
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor
import warnings
from typing import Optional, Tuple, Dict, Any

# Import from existing modules
from lib.merge.hash import nine_edge_hash, get_vc, nearest_neighbors


def calculate_scale_factor_from_positions(
    phenotype_positions: pd.DataFrame, sbs_positions: pd.DataFrame
) -> float:
    """Calculate scale factor by comparing coordinate ranges from cell positions.

    Args:
        phenotype_positions: High-resolution phenotype cell positions
        sbs_positions: Lower-resolution SBS cell positions

    Returns:
        Scale factor (SBS_range / phenotype_range)
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

    # Validate the scale factor is reasonable (should be around 0.25 for 40x->10x)
    if not (0.1 <= scale_factor <= 0.5):
        print(
            f"Warning: Unusual scale factor {scale_factor:.3f}. Expected ~0.25 for 40x->10x"
        )

    return scale_factor


def scale_coordinates(positions_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    """Scale coordinate columns by a factor.

    Args:
        positions_df: DataFrame with 'i' and 'j' coordinate columns
        scale_factor: Factor to multiply coordinates by

    Returns:
        DataFrame with scaled coordinates
    """
    scaled_df = positions_df.copy()
    scaled_df["i"] = scaled_df["i"] * scale_factor
    scaled_df["j"] = scaled_df["j"] * scale_factor
    return scaled_df


def calculate_coordinate_overlap(
    positions_1: pd.DataFrame, positions_2: pd.DataFrame
) -> float:
    """Calculate the overlap fraction between two coordinate datasets.

    Args:
        positions_1: First coordinate dataset
        positions_2: Second coordinate dataset

    Returns:
        Overlap fraction (0.0 to 1.0)
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
    """Generate triangle hash for well-level data using the exact same approach as tile-by-tile.

    This replicates the proven find_triangles function but for stitched well data.
    """
    if len(positions_df) < 4:
        return pd.DataFrame()

    coords = positions_df[["i", "j"]].values

    try:
        dt = Delaunay(coords)
        vectors, centers = [], []

        for i in range(dt.simplices.shape[0]):
            # Skip triangles with an edge on the outer boundary (same as tile-by-tile)
            if (dt.neighbors[i] == -1).any():
                continue

            # Use the exact same nine_edge_hash function from your working pipeline
            result = nine_edge_hash(dt, i)
            if result is None:
                continue

            _, v = result
            c = coords[dt.simplices[i], :].mean(axis=0)
            vectors.append(v)
            centers.append(c)

        # Convert to same format as tile-by-tile find_triangles output
        vectors_array = np.array(vectors).reshape(-1, 18)
        centers_array = np.array(centers)

        # Create DataFrame in exact same format as tile-by-tile approach
        df_vectors = pd.DataFrame(vectors_array).rename(columns=lambda x: f"V_{x}")
        df_coords = pd.DataFrame(centers_array).rename(columns=lambda x: f"c_{x}")
        df_combined = pd.concat([df_vectors, df_coords], axis=1)

        # Add magnitude column (critical for normalization)
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
    """Evaluate match using SINGLE fixed seed (42) for reproducible, fast results."""
    V_0, c_0 = get_vc(vec_centers_0)
    V_1, c_1 = get_vc(vec_centers_1)
    i0, i1, distances = nearest_neighbors(V_0, V_1)

    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]

    if sum(filt) < 5:
        return None, None, -1

    # Use ONLY seed 42 for fast, reproducible results
    print(f"Using single RANSAC seed (42) for fast evaluation...")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(
                random_state=42,  # Single fixed seed instead of 40 different seeds
                min_samples=max(5, len(X) // 10),
                max_trials=1000,
            )
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_
        determinant = np.linalg.det(rotation)

        # Calculate score
        distances = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances.min(axis=0)) < threshold_region
        score = (np.sqrt(distances.min(axis=0))[filt_score] < threshold_point).mean()

        print(f"✅ Single seed (42) result: det={determinant:.6f}, score={score:.3f}")

        return rotation, translation, score

    except Exception as e:
        print(f"❌ RANSAC with seed 42 failed: {e}")
        return None, None, -1


def geographic_constrained_sampling(
    cell_positions: pd.DataFrame,
    max_cells: int = 75000,
    center_radius: float = 0.4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample cells from the CENTER region for stable triangle hash alignment."""
    if len(cell_positions) <= max_cells:
        return cell_positions

    # Set fixed seed for reproducibility
    np.random.seed(random_state)

    print(f"=== Center Sampling (Seed: {random_state}) ===")
    print(f"Original cells: {len(cell_positions):,}")
    print(f"Target total: {max_cells:,}")
    print(f"Sampling CENTER {center_radius:.0%} region")

    # Calculate well boundaries and center
    i_coords = cell_positions["i"].values
    j_coords = cell_positions["j"].values

    i_min, i_max = i_coords.min(), i_coords.max()
    j_min, j_max = j_coords.min(), j_coords.max()
    i_range = i_max - i_min
    j_range = j_max - j_min

    center_i = (i_min + i_max) / 2
    center_j = (j_min + j_max) / 2

    # Calculate circular distance for each dataset independently
    well_radius = max(i_range, j_range) / 2

    # Add distance from center
    df = cell_positions.copy()
    df["dist_from_center_pixels"] = np.sqrt(
        (df["i"] - center_i) ** 2 + (df["j"] - center_j) ** 2
    )

    # Convert to normalized distance (0 = center, 1 = edge of well)
    df["dist_from_center_norm"] = df["dist_from_center_pixels"] / well_radius

    # Take cells from CENTER region
    center_mask = df["dist_from_center_norm"] <= center_radius
    center_cells = df[center_mask]

    print(f"Cells in center region (0%-{center_radius:.0%}): {len(center_cells):,}")

    if len(center_cells) == 0:
        print("❌ No cells found in center region! Using all cells")
        center_cells = df

    # Sample from center region
    if len(center_cells) > max_cells:
        center_sample = center_cells.nsmallest(max_cells, "dist_from_center_norm")
    else:
        center_sample = center_cells

    # Remove temporary columns
    final_sample = center_sample.drop(
        columns=["dist_from_center_pixels", "dist_from_center_norm"], errors="ignore"
    )

    print(f"Final sampled: {len(final_sample):,} cells")

    return final_sample


def sample_region_for_alignment(
    phenotype_resized: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    region_size: int = 7000,
    strategy: str = "center",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Sample a centered region from both datasets for triangle hash alignment.
    Uses coordinate-based approach like your visualization code.

    Args:
        phenotype_resized: Resized phenotype positions
        sbs_positions: SBS positions
        region_size: Size of the square region to sample
        strategy: Sampling strategy ("center" for now)

    Returns:
        Tuple of (phenotype_region, sbs_region, region_info)
    """
    print(f"=== Regional Sampling for Alignment ===")
    print(f"Region size: {region_size}x{region_size}")
    print(f"Strategy: {strategy}")

    # Find overlap bounds between the two datasets
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

    overlap_i_range = overlap_i_max - overlap_i_min
    overlap_j_range = overlap_j_max - overlap_j_min

    print(
        f"Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]"
    )
    print(f"Overlap size: {overlap_i_range:.0f} x {overlap_j_range:.0f}")

    if overlap_i_range <= 0 or overlap_j_range <= 0:
        print("❌ No overlap found between datasets")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Center the sampling region within the overlap (like visualization code)
    center_i = (overlap_i_min + overlap_i_max) / 2
    center_j = (overlap_j_min + overlap_j_max) / 2

    half_size = region_size / 2

    # Define sampling region bounds
    sample_i_min = max(overlap_i_min, center_i - half_size)
    sample_i_max = min(overlap_i_max, center_i + half_size)
    sample_j_min = max(overlap_j_min, center_j - half_size)
    sample_j_max = min(overlap_j_max, center_j + half_size)

    print(
        f"Sampling region: i=[{sample_i_min:.0f}, {sample_i_max:.0f}], j=[{sample_j_min:.0f}, {sample_j_max:.0f}]"
    )

    # Sample cells within the region
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

    print(f"Sampled cells: {len(pheno_region):,} phenotype, {len(sbs_region):,} SBS")

    region_info = {
        "region_size": region_size,
        "strategy": strategy,
        "bounds": {
            "i_min": sample_i_min,
            "i_max": sample_i_max,
            "j_min": sample_j_min,
            "j_max": sample_j_max,
        },
        "overlap_bounds": {
            "i_min": overlap_i_min,
            "i_max": overlap_i_max,
            "j_min": overlap_j_min,
            "j_max": overlap_j_max,
        },
        "cell_counts": {"phenotype": len(pheno_region), "sbs": len(sbs_region)},
    }

    return pheno_region, sbs_region, region_info


def triangle_hash_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    max_cells_for_hash: int = 75000,
    threshold_triangle: float = 0.1,
    threshold_point: float = 2.0,
    min_score: float = 0.05,
    adaptive_region: bool = True,
    initial_region_size: int = 7000,
    min_triangles: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """Triangle hash alignment for well-level data with adaptive regional sampling.

    Args:
        phenotype_positions: Phenotype cell positions (should be pre-resized)
        sbs_positions: SBS cell positions
        max_cells_for_hash: Maximum cells to use for triangle generation
        threshold_triangle: Triangle similarity threshold
        threshold_point: Point distance threshold
        min_score: Minimum score to accept alignment
        adaptive_region: Use adaptive regional sampling (default True)
        initial_region_size: Starting region size for sampling
        min_triangles: Minimum triangles needed for good alignment

    Returns:
        DataFrame with alignment parameters or empty DataFrame if failed
    """
    print(
        f"Triangle hash alignment with {len(phenotype_positions):,} phenotype and {len(sbs_positions):,} SBS cells"
    )

    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()

    if not adaptive_region:
        # Original approach - sample from full datasets
        print("Using full-well sampling approach")
        return _triangle_hash_full_well(
            phenotype_positions,
            sbs_positions,
            max_cells_for_hash,
            threshold_triangle,
            threshold_point,
            min_score,
        )

    # Adaptive regional approach
    print("Using adaptive regional sampling approach")

    region_size = initial_region_size
    max_region_size = (
        min(
            phenotype_positions["i"].max() - phenotype_positions["i"].min(),
            phenotype_positions["j"].max() - phenotype_positions["j"].min(),
            sbs_positions["i"].max() - sbs_positions["i"].min(),
            sbs_positions["j"].max() - sbs_positions["j"].min(),
        )
        * 0.8
    )  # Use 80% of smallest dimension as max

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts and region_size <= max_region_size:
        attempts += 1
        print(f"\n--- Attempt {attempts}: Region size {region_size:.0f} ---")

        # Sample region
        pheno_region, sbs_region, region_info = sample_region_for_alignment(
            phenotype_positions, sbs_positions, region_size=int(region_size)
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
        rotation, translation, score = evaluate_well_match(
            pheno_triangles,
            sbs_triangles,
            threshold_triangle=threshold_triangle,
            threshold_point=threshold_point,
        )

        if rotation is None or score < min_score:
            print(
                f"Triangle hash match failed: score={score:.3f} < {min_score}, increasing region size"
            )
            region_size *= 1.5
            continue

        # Success!
        determinant = np.linalg.det(rotation)

        print(f"✅ Regional triangle hash alignment successful:")
        print(f"   Score: {score:.3f}")
        print(f"   Determinant: {determinant:.6f}")
        print(f"   Region size used: {region_size:.0f}")

        # Build result
        alignment = {
            "rotation": rotation,
            "translation": translation,
            "score": score,
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

    return pd.DataFrame()


def _triangle_hash_full_well(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    max_cells_for_hash: int,
    threshold_triangle: float,
    threshold_point: float,
    min_score: float,
) -> pd.DataFrame:
    """Original full-well triangle hash approach (preserved for fallback)."""
    # Sample cells if datasets are too large
    if len(phenotype_positions) > max_cells_for_hash:
        pheno_subset = geographic_constrained_sampling(
            phenotype_positions,
            max_cells=max_cells_for_hash,
            center_radius=0.4,
            random_state=42,
        )
    else:
        pheno_subset = phenotype_positions.copy()

    if len(sbs_positions) > max_cells_for_hash:
        sbs_subset = geographic_constrained_sampling(
            sbs_positions,
            max_cells=max_cells_for_hash,
            center_radius=0.4,
            random_state=42,
        )
    else:
        sbs_subset = sbs_positions.copy()

    # Generate triangle hashes
    pheno_triangles = well_level_triangle_hash(pheno_subset)
    sbs_triangles = well_level_triangle_hash(sbs_subset)

    if len(pheno_triangles) == 0 or len(sbs_triangles) == 0:
        print("Failed to generate triangle hashes")
        return pd.DataFrame()

    print(
        f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles"
    )

    # Evaluate triangle hash match
    rotation, translation, score = evaluate_well_match(
        pheno_triangles,
        sbs_triangles,
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point,
    )

    if rotation is None or score < min_score:
        print(f"Triangle hash match failed: score={score:.3f} < {min_score}")
        return pd.DataFrame()

    determinant = np.linalg.det(rotation)

    print(f"✅ Triangle hash alignment successful:")
    print(f"   Score: {score:.3f}")
    print(f"   Determinant: {determinant:.6f}")

    # Build result
    alignment = {
        "rotation": rotation,
        "translation": translation,
        "score": score,
        "determinant": determinant,
        "transformation_type": "triangle_hash_well_level",
        "triangles_matched": len(pheno_triangles),
        "approach": "triangle_hash_after_scaling",
    }

    return pd.DataFrame([alignment])
