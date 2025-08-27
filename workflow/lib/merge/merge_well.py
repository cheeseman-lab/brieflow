"""Well-level merge functions for stitched image data.

This module provides functions for merging phenotype and SBS cell data at the well level,
using triangle hashing for alignment and robust cell matching. It includes multiple
alignment strategies including triangle hash matching and hardcoded scaling approaches.
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
from typing import Optional, Tuple, Dict, Any, Union
from lib.merge.hash import nine_edge_hash, get_vc, nearest_neighbors


def well_level_triangle_hash(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Generate triangle hash vectors for well-level cell positions.

    Creates triangle hash vectors using Delaunay triangulation of cell positions.
    This approach mirrors the tile-by-tile triangle hashing methodology but adapted
    for well-level stitched data.

    Args:
        positions_df: DataFrame containing cell positions with 'i' and 'j' columns

    Returns:
        DataFrame containing triangle vectors, centers, and magnitudes.
        Returns empty DataFrame if triangulation fails or insufficient points.
    """
    if len(positions_df) < 4:
        print("Insufficient cells for triangulation (need at least 4)")
        return pd.DataFrame()

    coords = positions_df[["i", "j"]].values

    try:
        dt = Delaunay(coords)
        vectors, centers = [], []

        for i in range(dt.simplices.shape[0]):
            # Skip boundary triangles to avoid edge artifacts
            if (dt.neighbors[i] == -1).any():
                continue

            result = nine_edge_hash(dt, i)
            if result is None:
                continue

            _, v = result
            c = coords[dt.simplices[i], :].mean(axis=0)
            vectors.append(v)
            centers.append(c)

        if not vectors:
            print("No valid triangles found after filtering boundary triangles")
            return pd.DataFrame()

        # Convert to standardized format
        vectors_array = np.array(vectors).reshape(-1, 18)
        centers_array = np.array(centers)

        df_vectors = pd.DataFrame(vectors_array).rename(columns=lambda x: f"V_{x}")
        df_coords = pd.DataFrame(centers_array).rename(columns=lambda x: f"c_{x}")
        df_combined = pd.concat([df_vectors, df_coords], axis=1)

        # Add magnitude for normalization
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
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Evaluate triangle hash match between two datasets using RANSAC.

    Uses a single fixed random seed for reproducible and fast results, rather than
    multiple seeds which would be computationally expensive.

    Args:
        vec_centers_0: Triangle vectors and centers from first dataset
        vec_centers_1: Triangle vectors and centers from second dataset
        threshold_triangle: Maximum distance for triangle similarity matching
        threshold_point: Maximum distance for point matching in pixels
        random_state: Fixed random seed for reproducible results

    Returns:
        Tuple of (rotation_matrix, translation_vector, match_score).
        Returns (None, None, -1) if matching fails.
    """
    V_0, c_0 = get_vc(vec_centers_0)
    V_1, c_1 = get_vc(vec_centers_1)
    i0, i1, distances = nearest_neighbors(V_0, V_1)

    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]

    if sum(filt) < 5:
        print(f"Insufficient triangle matches: {sum(filt)} (need at least 5)")
        return None, None, -1

    print(f"Using RANSAC with fixed seed ({random_state}) for reproducible results")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(
                random_state=random_state,
                min_samples=max(5, len(X) // 10),
                max_trials=1000,
            )
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_
        determinant = np.linalg.det(rotation)

        # Calculate match quality score
        distances = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances.min(axis=0)) < threshold_region
        score = (np.sqrt(distances.min(axis=0))[filt_score] < threshold_point).mean()

        print(f"Match evaluation: determinant={determinant:.6f}, score={score:.3f}")

        return rotation, translation, score

    except Exception as e:
        print(f"RANSAC matching failed: {e}")
        return None, None, -1


def triangle_hash_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    max_cells_for_hash: int = 75000,
    threshold_triangle: float = 0.1,
    threshold_point: float = 2.0,
    min_score: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """Perform triangle hash-based alignment for well-level data.

    This function creates triangle hashes from both datasets and uses them to find
    the optimal transformation between phenotype and SBS coordinate systems.

    Args:
        phenotype_positions: Phenotype cell positions (should be pre-scaled if needed)
        sbs_positions: SBS cell positions
        max_cells_for_hash: Maximum cells to use for triangle generation
        threshold_triangle: Triangle similarity threshold for matching
        threshold_point: Point distance threshold for validation
        min_score: Minimum match score to accept alignment
        **kwargs: Additional arguments from calling functions

    Returns:
        DataFrame with alignment parameters including rotation, translation, and
        quality metrics. Returns empty DataFrame if alignment fails.
    """
    print(f"Triangle hash alignment: {len(phenotype_positions):,} phenotype, "
          f"{len(sbs_positions):,} SBS cells")

    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()

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

    print(f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles")

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

    print(f"Triangle hash alignment successful:")
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


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    threshold: float = 10.0,
    chunk_size: int = 50000,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Merge cells using alignment transformation with memory-efficient processing.

    Applies the computed transformation to phenotype coordinates and finds the closest
    SBS cells within the specified threshold distance.

    Args:
        phenotype_positions: DataFrame with phenotype cell positions
        sbs_positions: DataFrame with SBS cell positions
        alignment: Alignment parameters (rotation matrix and translation vector)
        threshold: Maximum distance threshold for cell matching in pixels
        chunk_size: Number of cells to process per chunk for memory efficiency
        output_path: Optional path to save raw matches before deduplication

    Returns:
        DataFrame with merged cell pairs, deduplicated to ensure one-to-one matching.
    """
    print(f"Starting cell merging with threshold={threshold}")

    # Extract transformation parameters
    if isinstance(alignment, pd.Series):
        rotation = alignment["rotation"]
        translation = alignment["translation"]
    elif isinstance(alignment, dict):
        rotation = alignment.get("rotation", np.eye(2))
        translation = alignment.get("translation", np.zeros(2))
    else:
        rotation = (
            alignment.iloc[0]["rotation"]
            if "rotation" in alignment.columns
            else np.eye(2)
        )
        translation = (
            alignment.iloc[0]["translation"]
            if "translation" in alignment.columns
            else np.zeros(2)
        )

    # Ensure valid shapes
    rotation = (
        np.array(rotation).reshape(2, 2)
        if rotation is not None and np.array(rotation).size == 4
        else np.eye(2)
    )
    translation = (
        np.array(translation).flatten()[:2]
        if translation is not None and np.array(translation).size >= 2
        else np.zeros(2)
    )

    print(f"Transformation: rotation determinant={np.linalg.det(rotation):.3f}, "
          f"translation={translation}")

    # Prepare coordinates
    pheno_coords = phenotype_positions[["i", "j"]].to_numpy()
    sbs_coords = sbs_positions[["i", "j"]].to_numpy()

    # Transform phenotype coordinates to SBS coordinate space
    transformed_coords = pheno_coords @ rotation.T + translation

    print(f"Coordinate ranges after transformation:")
    print(f"  Transformed phenotype: i=[{transformed_coords[:, 0].min():.0f}, "
          f"{transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, "
          f"{transformed_coords[:, 1].max():.0f}]")
    print(f"  SBS: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], "
          f"j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")

    # Pre-extract metadata arrays for safe indexing
    pheno_cells = phenotype_positions["cell"].to_numpy()
    pheno_i = phenotype_positions["i"].to_numpy()
    pheno_j = phenotype_positions["j"].to_numpy()
    pheno_area = (
        phenotype_positions["area"].to_numpy()
        if "area" in phenotype_positions.columns
        else np.full(len(pheno_cells), np.nan)
    )

    sbs_cells = sbs_positions["cell"].to_numpy()
    sbs_i = sbs_positions["i"].to_numpy()
    sbs_j = sbs_positions["j"].to_numpy()
    sbs_area = (
        sbs_positions["area"].to_numpy()
        if "area" in sbs_positions.columns
        else np.full(len(sbs_cells), np.nan)
    )

    # Process in chunks to manage memory
    all_matches = []
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size
    print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks of {chunk_size:,}")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))

        if chunk_idx % 10 == 0:
            print(f"Processing chunk {chunk_idx + 1}/{n_chunks}")

        sbs_chunk_coords = sbs_coords[start_idx:end_idx]
        if sbs_chunk_coords.size == 0:
            continue

        distances = cdist(sbs_chunk_coords, transformed_coords, metric="euclidean")
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances[np.arange(distances.shape[0]), closest_pheno_idx]

        valid_mask = min_distances < threshold
        if not valid_mask.any():
            continue

        sbs_idx_global = np.nonzero(valid_mask)[0] + start_idx
        pheno_idx = closest_pheno_idx[valid_mask]
        match_distances = min_distances[valid_mask]

        chunk_matches = pd.DataFrame(
            {
                "cell_0": pheno_cells[pheno_idx],
                "i_0": pheno_i[pheno_idx],
                "j_0": pheno_j[pheno_idx],
                "cell_1": sbs_cells[sbs_idx_global],
                "i_1": sbs_i[sbs_idx_global],
                "j_1": sbs_j[sbs_idx_global],
                "distance": match_distances,
                "area_0": pheno_area[pheno_idx],
                "area_1": sbs_area[sbs_idx_global],
            }
        )

        all_matches.append(chunk_matches)

    # Combine and deduplicate
    if not all_matches:
        print("No cells matched within threshold")
        return pd.DataFrame(
            columns=[
                "cell_0", "i_0", "j_0", "area_0",
                "cell_1", "i_1", "j_1", "area_1", "distance"
            ]
        )

    merged_cells_raw = pd.concat(all_matches, ignore_index=True)
    print(f"Before deduplication: {len(merged_cells_raw):,} matches")
    print(f"Duplicate phenotype cells: {merged_cells_raw['cell_0'].duplicated().sum():,}")

    # Save raw matches if requested
    if output_path is not None:
        try:
            output_path_str = str(
                getattr(output_path, "__fspath__", lambda: str(output_path))()
            )
            raw_matches_path = output_path_str.replace(".parquet", "_raw_matches.parquet")
            merged_cells_raw.to_parquet(raw_matches_path)
            print(f"Saved raw matches to: {raw_matches_path}")
        except Exception as e:
            print(f"Could not save raw matches: {e}")

    # Deduplicate by keeping best match for each phenotype cell
    merged_cells = merged_cells_raw.sort_values("distance").drop_duplicates(
        "cell_0", keep="first"
    )
    
    print(f"After deduplication: {len(merged_cells):,} matches")
    print(f"Successfully merged {len(merged_cells):,} cells")
    print(f"Distance statistics: mean={merged_cells['distance'].mean():.2f}, "
          f"max={merged_cells['distance'].max():.2f}")

    return merged_cells


def geographic_constrained_sampling(
    cell_positions: pd.DataFrame,
    max_cells: int = 75000,
    center_radius: float = 0.4,
    random_state: int = 42,
    **kwargs  # Ignore unused parameters for backwards compatibility
) -> pd.DataFrame:
    """Sample cells from the center region of the well for stable alignment.

    Center cells provide more stable triangulation compared to edge cells, which can
    be affected by boundary artifacts and imaging variations.

    Args:
        cell_positions: DataFrame with cell positions containing 'i' and 'j' columns
        max_cells: Maximum number of cells to sample
        center_radius: Fraction of well radius defining center region (0.4 = 40%)
        random_state: Random seed for reproducible sampling
        **kwargs: Additional unused parameters for backwards compatibility

    Returns:
        DataFrame with sampled cell positions from the center region.
    """
    if len(cell_positions) <= max_cells:
        return cell_positions

    np.random.seed(random_state)

    print(f"Geographic sampling from center region (seed: {random_state})")
    print(f"Original cells: {len(cell_positions):,}, Target: {max_cells:,}")

    # Calculate well boundaries and center
    i_coords = cell_positions["i"].values
    j_coords = cell_positions["j"].values

    i_min, i_max = i_coords.min(), i_coords.max()
    j_min, j_max = j_coords.min(), j_coords.max()
    i_range = i_max - i_min
    j_range = j_max - j_min

    center_i = (i_min + i_max) / 2
    center_j = (j_min + j_max) / 2

    print(f"Dataset boundaries: i=[{i_min:.0f}, {i_max:.0f}], "
          f"j=[{j_min:.0f}, {j_max:.0f}]")
    print(f"Dataset center: ({center_i:.0f}, {center_j:.0f})")

    # Calculate distance from center
    well_radius = max(i_range, j_range) / 2
    df = cell_positions.copy()
    df["dist_from_center_pixels"] = np.sqrt(
        (df["i"] - center_i) ** 2 + (df["j"] - center_j) ** 2
    )
    df["dist_from_center_norm"] = df["dist_from_center_pixels"] / well_radius

    print(f"Sampling from center {center_radius:.0%} region")

    # Sample from center region
    center_mask = df["dist_from_center_norm"] <= center_radius
    center_cells = df[center_mask]

    print(f"Cells in center region: {len(center_cells):,} "
          f"({100 * len(center_cells) / len(cell_positions):.1f}% of total)")

    if len(center_cells) == 0:
        print("Warning: No cells found in center region, expanding search")
        wider_radius = 0.6
        center_mask_wider = df["dist_from_center_norm"] <= wider_radius
        center_cells = df[center_mask_wider]
        print(f"Expanded to {wider_radius:.0%}: {len(center_cells):,} cells")

        if len(center_cells) == 0:
            print("Warning: Still no cells found, using all cells")
            center_cells = df

    # Sample requested number of cells, prioritizing cells closest to center
    if len(center_cells) > max_cells:
        center_sample = center_cells.nsmallest(max_cells, "dist_from_center_norm")
    else:
        center_sample = center_cells

    # Remove temporary columns
    final_sample = center_sample.drop(
        columns=["dist_from_center_pixels", "dist_from_center_norm"], errors="ignore"
    )

    print(f"Final sample: {len(final_sample):,} cells from center region")

    return final_sample


def pure_scaling_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: Optional[float] = None,
    validation_sample_size: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Perform pure scaling transformation with no translation.

    Tests if coordinate systems are already aligned and only need scaling to match.
    This is useful when the imaging systems have different pixel sizes but are
    otherwise perfectly aligned.

    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions  
        scale_factor: Fixed scale factor (if None, calculated from coordinate ranges)
        validation_sample_size: Number of cells for validation
        random_state: Random seed for reproducible sampling

    Returns:
        DataFrame with alignment parameters including validation metrics.
    """
    print(f"Pure scaling alignment (no translation)")

    pheno_coords = phenotype_positions[["i", "j"]].values
    sbs_coords = sbs_positions[["i", "j"]].values

    # Calculate coordinate ranges
    pheno_i_range = np.ptp(pheno_coords[:, 0])
    pheno_j_range = np.ptp(pheno_coords[:, 1])
    sbs_i_range = np.ptp(sbs_coords[:, 0])
    sbs_j_range = np.ptp(sbs_coords[:, 1])

    print(f"Coordinate ranges:")
    print(f"  Phenotype: i={pheno_i_range:.0f}, j={pheno_j_range:.0f}")
    print(f"  SBS: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")

    # Calculate or use provided scale factor
    if scale_factor is None:
        scale_i = sbs_i_range / pheno_i_range
        scale_j = sbs_j_range / pheno_j_range
        scale_factor = (scale_i + scale_j) / 2

        print(f"Calculated scale factors:")
        print(f"  i-direction: {scale_i:.6f}")
        print(f"  j-direction: {scale_j:.6f}")
        print(f"  Average: {scale_factor:.6f}")
    else:
        print(f"Using provided scale factor: {scale_factor:.6f}")

    # Build transformation matrices
    rotation_matrix = np.array([[scale_factor, 0.0], [0.0, scale_factor]])
    translation = np.array([0.0, 0.0])

    print(f"Pure scaling transformation (no translation)")

    # Validate transformation
    np.random.seed(random_state)
    
    if len(phenotype_positions) > validation_sample_size:
        pheno_sample_idx = np.random.choice(
            len(phenotype_positions), validation_sample_size, replace=False
        )
        pheno_sample = pheno_coords[pheno_sample_idx]
    else:
        pheno_sample = pheno_coords

    if len(sbs_positions) > validation_sample_size:
        sbs_sample_idx = np.random.choice(
            len(sbs_positions), validation_sample_size, replace=False
        )
        sbs_sample = sbs_coords[sbs_sample_idx]
    else:
        sbs_sample = sbs_coords

    # Apply scaling transformation
    transformed_pheno = pheno_sample * scale_factor

    # Calculate overlap
    overlap_i_min = max(transformed_pheno[:, 0].min(), sbs_sample[:, 0].min())
    overlap_i_max = min(transformed_pheno[:, 0].max(), sbs_sample[:, 0].max())
    overlap_j_min = max(transformed_pheno[:, 1].min(), sbs_sample[:, 1].min())
    overlap_j_max = min(transformed_pheno[:, 1].max(), sbs_sample[:, 1].max())

    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min

    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = (sbs_sample[:, 0].max() - sbs_sample[:, 0].min()) * (
            sbs_sample[:, 1].max() - sbs_sample[:, 1].min()
        )
        overlap_fraction = overlap_area / sbs_area if sbs_area > 0 else 0
    else:
        overlap_fraction = 0.0

    # Validate alignment quality
    distances = cdist(transformed_pheno, sbs_sample, metric="euclidean")
    min_distances = distances.min(axis=1)

    # Calculate scores at different thresholds
    thresholds = [2, 5, 10, 20, 50]
    scores = {}
    for thresh in thresholds:
        good_matches = (min_distances < thresh).sum()
        score = good_matches / len(min_distances)
        scores[thresh] = score

    mean_distance = min_distances.mean()
    median_distance = np.median(min_distances)

    print(f"Validation results:")
    print(f"  Overlap fraction: {overlap_fraction:.1%}")
    print(f"  Mean distance: {mean_distance:.2f} px")
    print(f"  Score (10px threshold): {scores[10]:.3f}")

    # Build result
    alignment = {
        "rotation": rotation_matrix,
        "translation": translation,
        "score": scores[10],
        "determinant": np.linalg.det(rotation_matrix),
        "transformation_type": "pure_scaling_no_translation",
        "scale_factor": scale_factor,
        "approach": "coordinate_range_scaling",
        "overlap_fraction": overlap_fraction,
        "validation_mean_distance": mean_distance,
        "validation_median_distance": median_distance,
        "scores_by_threshold": scores,
        "has_overlap": has_overlap,
    }

    return pd.DataFrame([alignment])


def hardcoded_scale_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = 0.125,
    max_cells_for_translation: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Use hardcoded scale factor and learn translation from data.

    This approach is useful when the scale factor is known (e.g., from instrument
    specifications) but the translation between coordinate systems needs to be
    determined from the data.

    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions
        scale_factor: Fixed scale factor (default 0.125 = 1/8 for common setups)
        max_cells_for_translation: Maximum cells for translation estimation
        random_state: Random seed for reproducible results

    Returns:
        DataFrame with alignment parameters.
    """
    print(f"Hardcoded scale + learned translation")
    print(f"Using fixed scale factor: {scale_factor}")

    # Sample cells for translation learning
    if len(phenotype_positions) > max_cells_for_translation:
        pheno_sample = geographic_constrained_sampling(
            phenotype_positions,
            max_cells=max_cells_for_translation,
            center_radius=0.3,
            random_state=random_state,
        )
    else:
        pheno_sample = phenotype_positions.copy()

    if len(sbs_positions) > max_cells_for_translation:
        sbs_sample = geographic_constrained_sampling(
            sbs_positions,
            max_cells=max_cells_for_translation,
            center_radius=0.3,
            random_state=random_state,
        )
    else:
        sbs_sample = sbs_positions.copy()

    # Build hardcoded transformation matrix
    rotation_matrix = np.array([[scale_factor, 0.0], [0.0, scale_factor]])

    # Learn translation using center of mass alignment
    pheno_coords = pheno_sample[["i", "j"]].values
    sbs_coords = sbs_sample[["i", "j"]].values

    # Scale phenotype coordinates
    scaled_pheno_coords = pheno_coords * scale_factor

    # Calculate centers of mass
    pheno_center = scaled_pheno_coords.mean(axis=0)
    sbs_center = sbs_coords.mean(axis=0)

    # Translation = difference between centers
    translation = sbs_center - pheno_center

    print(f"Center of mass translation: [{translation[0]:.1f}, {translation[1]:.1f}]")

    # Validate the transformation
    transformed_pheno = scaled_pheno_coords + translation

    distances = cdist(transformed_pheno, sbs_coords, metric="euclidean")
    min_distances = distances.min(axis=1)

    mean_distance = min_distances.mean()
    good_matches = (min_distances < 10.0).sum()
    score = good_matches / len(min_distances)

    print(f"Validation: {good_matches:,}/{len(min_distances):,} matches within 10px ({score:.1%})")

    # Build result
    alignment = {
        "rotation": rotation_matrix,
        "translation": translation,
        "score": score,
        "determinant": np.linalg.det(rotation_matrix),
        "transformation_type": "hardcoded_scale_learned_translation",
        "scale_factor": scale_factor,
        "approach": "center_of_mass_translation",
        "validation_mean_distance": mean_distance,
        "validation_good_matches": good_matches,
    }

    return pd.DataFrame([alignment])


def hardcoded_scale_robust_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = 0.125,
    max_cells_for_translation: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Use hardcoded scale with robust translation estimation via RANSAC.

    This approach uses RANSAC to robustly estimate translation from point
    correspondences, making it more resistant to outliers than simple center
    of mass alignment.

    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions
        scale_factor: Fixed scale factor
        max_cells_for_translation: Maximum cells for translation estimation
        random_state: Random seed for reproducible results

    Returns:
        DataFrame with alignment parameters.
    """
    print(f"Hardcoded scale + robust RANSAC translation")

    # Sample cells for translation learning
    if len(phenotype_positions) > max_cells_for_translation:
        pheno_sample = geographic_constrained_sampling(
            phenotype_positions,
            max_cells=max_cells_for_translation,
            center_radius=0.4,
            random_state=random_state,
        )
    else:
        pheno_sample = phenotype_positions.copy()

    if len(sbs_positions) > max_cells_for_translation:
        sbs_sample = geographic_constrained_sampling(
            sbs_positions,
            max_cells=max_cells_for_translation,
            center_radius=0.4,
            random_state=random_state,
        )
    else:
        sbs_sample = sbs_positions.copy()

    # Build hardcoded scaling matrix
    rotation_matrix = np.array([[scale_factor, 0.0], [0.0, scale_factor]])

    # Scale phenotype coordinates
    pheno_coords = pheno_sample[["i", "j"]].values
    sbs_coords = sbs_sample[["i", "j"]].values
    scaled_pheno_coords = pheno_coords * scale_factor

    # Find initial correspondences using nearest neighbors
    distances = cdist(scaled_pheno_coords, sbs_coords, metric="euclidean")

    # For each scaled phenotype cell, find closest SBS cell
    closest_sbs_idx = distances.argmin(axis=1)
    closest_distances = distances.min(axis=1)

    # Filter to reasonable matches only
    reasonable_threshold = 50.0
    good_matches = closest_distances < reasonable_threshold

    if good_matches.sum() < 10:
        print(f"Too few reasonable initial matches: {good_matches.sum()}")
        return pd.DataFrame()

    # Get matched point pairs
    pheno_matched = scaled_pheno_coords[good_matches]
    sbs_matched = sbs_coords[closest_sbs_idx[good_matches]]

    print(f"Found {len(pheno_matched):,} initial point correspondences")

    # Use RANSAC to robustly estimate translation
    try:
        model = RANSACRegressor(
            random_state=random_state,
            min_samples=max(5, len(pheno_matched) // 20),
            max_trials=1000,
            residual_threshold=5.0,
        )

        model.fit(pheno_matched, sbs_matched)

        # Extract translation (intercept of linear model)
        translation = model.estimator_.intercept_

        # Validate the transformation
        all_scaled_pheno = pheno_coords * scale_factor
        transformed_coords = all_scaled_pheno + translation

        distances_all = cdist(transformed_coords, sbs_coords, metric="euclidean")
        min_distances_all = distances_all.min(axis=1)

        mean_distance = min_distances_all.mean()
        good_matches_count = (min_distances_all < 10.0).sum()
        score = good_matches_count / len(min_distances_all)

        print(f"Robust translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
        print(f"Validation: {good_matches_count:,}/{len(min_distances_all):,} "
              f"matches within 10px ({score:.1%})")

        alignment = {
            "rotation": rotation_matrix,
            "translation": translation,
            "score": score,
            "determinant": np.linalg.det(rotation_matrix),
            "transformation_type": "hardcoded_scale_ransac_translation",
            "scale_factor": scale_factor,
            "approach": "robust_point_correspondence",
            "validation_mean_distance": mean_distance,
            "validation_good_matches": good_matches_count,
        }

        return pd.DataFrame([alignment])

    except Exception as e:
        print(f"Robust translation estimation failed: {e}")
        return pd.DataFrame()


def check_alignment_quality_permissive(
    alignment: pd.Series,
    det_range: list,
    score_threshold: float
) -> bool:
    """Check alignment quality with permissive criteria.

    Accepts alignment if it meets basic criteria for determinant bounds,
    score threshold, and has a positive determinant.

    Args:
        alignment: Alignment result series containing quality metrics
        det_range: [min_det, max_det] determinant bounds for acceptance
        score_threshold: Minimum score threshold for acceptance

    Returns:
        True if alignment meets all acceptance criteria.
    """
    det = alignment.get("determinant", 0)
    score = alignment.get("score", 0)

    # Check criteria
    det_ok = det_range[0] <= det <= det_range[1] if det_range else True
    score_ok = score >= score_threshold
    positive_det = det > 0

    print(f"Alignment quality check:")
    print(f"  Determinant: {det:.6f} (bounds: {det_range})")
    print(f"  Score: {score:.3f} (threshold: {score_threshold})")
    print(f"  Positive determinant: {positive_det}")

    all_criteria_met = det_ok and score_ok and positive_det

    if all_criteria_met:
        print("  Result: ACCEPTED - all criteria met")
    else:
        print("  Result: REJECTED - criteria not met:")
        if not positive_det:
            print("    - Determinant not positive")
        if not det_ok:
            print("    - Determinant outside bounds")
        if not score_ok:
            print("    - Score below threshold")

    return all_criteria_met


# Legacy compatibility functions
def calculate_expected_scale_factor(
    phenotype_pixel_size: Optional[float],
    sbs_pixel_size: Optional[float]
) -> Optional[float]:
    """Calculate expected scale factor from pixel sizes.

    This is a diagnostic function that calculates the theoretical scale factor
    based on the pixel sizes of the two imaging systems.

    Args:
        phenotype_pixel_size: Pixel size in microns for phenotype imaging
        sbs_pixel_size: Pixel size in microns for SBS imaging

    Returns:
        Expected scale factor (phenotype/sbs) or None if pixel sizes not provided.
    """
    if phenotype_pixel_size and sbs_pixel_size:
        expected_scale = phenotype_pixel_size / sbs_pixel_size
        return expected_scale
    return None


def build_linear_model(
    rotation: np.ndarray,
    translation: np.ndarray
) -> LinearRegression:
    """Build a linear regression model using provided transformation parameters.

    This creates a scikit-learn LinearRegression model with the transformation
    parameters, useful for compatibility with legacy code.

    Args:
        rotation: 2x2 rotation/scaling matrix
        translation: 2-element translation vector

    Returns:
        LinearRegression model with coefficients and intercept set.
    """
    model = LinearRegression()
    model.coef_ = rotation
    model.intercept_ = translation
    return model


def merge_sbs_phenotype(
    cell_locations_0: pd.DataFrame,
    cell_locations_1: pd.DataFrame,
    model: LinearRegression,
    threshold: float = 2
) -> pd.DataFrame:
    """Legacy function for backwards compatibility with tile-by-tile approach.

    This function maintains compatibility with older tile-based merging code.

    Args:
        cell_locations_0: First dataset cell locations
        cell_locations_1: Second dataset cell locations  
        model: Linear model containing transformation parameters
        threshold: Distance threshold for matching

    Returns:
        DataFrame with merged cell pairs in legacy format.
    """
    cols_final = [
        "plate", "well", "tile", "cell_0", "i_0", "j_0", 
        "site", "cell_1", "i_1", "j_1", "distance"
    ]

    if (cell_locations_0 is None or cell_locations_1 is None or
        (hasattr(cell_locations_0, "empty") and cell_locations_0.empty) or
        (hasattr(cell_locations_1, "empty") and cell_locations_1.empty)):
        return pd.DataFrame(columns=cols_final)

    X = cell_locations_0[["i", "j"]].values
    Y = cell_locations_1[["i", "j"]].values
    Y_pred = model.predict(X)
    distances = cdist(Y, Y_pred, metric="sqeuclidean")
    ix = distances.argmin(axis=1)
    filt = np.sqrt(distances.min(axis=1)) < threshold

    columns_0 = {"tile": "tile", "cell": "cell_0", "i": "i_0", "j": "j_0"}
    columns_1 = {"site": "site", "cell": "cell_1", "i": "i_1", "j": "j_1"}

    target = (
        cell_locations_0.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)
    )

    return (
        cell_locations_1[filt]
        .reset_index(drop=True)[list(columns_1.keys())]
        .rename(columns=columns_1)
        .pipe(lambda x: pd.concat([target, x], axis=1))
        .assign(distance=np.sqrt(distances.min(axis=1))[filt])[cols_final]
    )


# Legacy function redirects
def stitched_well_alignment(*args, **kwargs) -> pd.DataFrame:
    """Legacy function redirected to triangle hash approach."""
    print("Note: Redirecting to triangle hash approach")
    return triangle_hash_well_alignment(*args, **kwargs)


def enhanced_alignment_with_correct_scale(*args, **kwargs) -> pd.DataFrame:
    """Legacy function redirected to triangle hash approach."""
    print("Note: Using triangle hash approach (no hardcoded scale needed)")
    return triangle_hash_well_alignment(*args, **kwargs)


# Diagnostic functions
def debug_transformation_matrix(
    rotation: np.ndarray,
    translation: np.ndarray
) -> float:
    """Debug and analyze transformation matrix components.

    Prints detailed information about the transformation matrix including
    scale factors, rotation components, and whether it represents pure scaling.

    Args:
        rotation: 2x2 transformation matrix
        translation: 2-element translation vector

    Returns:
        Average scale factor extracted from the matrix.
    """
    print(f"\n=== Transformation Matrix Analysis ===")
    print(f"Rotation matrix:")
    print(f"  [{rotation[0, 0]:.6f}, {rotation[0, 1]:.6f}]")
    print(f"  [{rotation[1, 0]:.6f}, {rotation[1, 1]:.6f}]")
    print(f"Translation: [{translation[0]:.2f}, {translation[1]:.2f}]")

    # Calculate scale factors
    scale_x = np.linalg.norm(rotation[0, :])
    scale_y = np.linalg.norm(rotation[1, :])
    avg_scale = (scale_x + scale_y) / 2

    print(f"Scale factors extracted from matrix:")
    print(f"  X-direction: {scale_x:.6f}")
    print(f"  Y-direction: {scale_y:.6f}")
    print(f"  Average: {avg_scale:.6f}")

    # Check rotation component
    if scale_x > 0 and scale_y > 0:
        normalized_rotation = rotation / avg_scale
        print(f"Normalized rotation matrix (scale removed):")
        print(f"  [{normalized_rotation[0, 0]:.6f}, {normalized_rotation[0, 1]:.6f}]")
        print(f"  [{normalized_rotation[1, 0]:.6f}, {normalized_rotation[1, 1]:.6f}]")

        # Check if it's close to identity (no rotation)
        if np.allclose(normalized_rotation, np.eye(2), atol=0.1):
            print("Pure scaling transformation (no significant rotation)")
        else:
            print("Scaling + rotation transformation")

    return avg_scale


def debug_coordinate_transformation(
    pheno_coords: np.ndarray,
    sbs_coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray
) -> Tuple[bool, float]:
    """Debug coordinate transformation results and overlap.

    Analyzes the coordinate transformation and calculates overlap between
    the transformed phenotype coordinates and SBS coordinates.

    Args:
        pheno_coords: Original phenotype coordinates
        sbs_coords: SBS coordinates
        rotation: Transformation rotation matrix
        translation: Transformation translation vector

    Returns:
        Tuple of (has_overlap, overlap_fraction).
    """
    print(f"\n=== Coordinate Transformation Analysis ===")

    # Apply transformation
    transformed_coords = pheno_coords @ rotation.T + translation

    # Compare coordinate ranges
    print(f"Original phenotype range:")
    print(f"  i: [{pheno_coords[:, 0].min():.0f}, {pheno_coords[:, 0].max():.0f}] "
          f"(range: {pheno_coords[:, 0].max() - pheno_coords[:, 0].min():.0f})")
    print(f"  j: [{pheno_coords[:, 1].min():.0f}, {pheno_coords[:, 1].max():.0f}] "
          f"(range: {pheno_coords[:, 1].max() - pheno_coords[:, 1].min():.0f})")

    print(f"Transformed phenotype range:")
    print(f"  i: [{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}] "
          f"(range: {transformed_coords[:, 0].max() - transformed_coords[:, 0].min():.0f})")
    print(f"  j: [{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}] "
          f"(range: {transformed_coords[:, 1].max() - transformed_coords[:, 1].min():.0f})")

    print(f"SBS coordinate range:")
    print(f"  i: [{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}] "
          f"(range: {sbs_coords[:, 0].max() - sbs_coords[:, 0].min():.0f})")
    print(f"  j: [{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}] "
          f"(range: {sbs_coords[:, 1].max() - sbs_coords[:, 1].min():.0f})")

    # Calculate overlap
    overlap_i_min = max(transformed_coords[:, 0].min(), sbs_coords[:, 0].min())
    overlap_i_max = min(transformed_coords[:, 0].max(), sbs_coords[:, 0].max())
    overlap_j_min = max(transformed_coords[:, 1].min(), sbs_coords[:, 1].min())
    overlap_j_max = min(transformed_coords[:, 1].max(), sbs_coords[:, 1].max())

    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min
    overlap_fraction = 0.0

    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = (sbs_coords[:, 0].max() - sbs_coords[:, 0].min()) * (
            sbs_coords[:, 1].max() - sbs_coords[:, 1].min()
        )
        overlap_fraction = overlap_area / sbs_area if sbs_area > 0 else 0

        print(f"Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], "
              f"j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
        print(f"Overlap fraction: {overlap_fraction:.1%} of SBS area")
    else:
        print("NO OVERLAP between transformed phenotype and SBS coordinates")

    return has_overlap, overlap_fraction