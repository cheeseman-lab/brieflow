"""
Step 1 Library: Coordinate scaling, triangle hashing, and alignment functions.
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


def scale_coordinates(positions_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    """
    Scale coordinate columns by a factor.
    
    Args:
        positions_df: DataFrame with 'i' and 'j' coordinate columns
        scale_factor: Factor to multiply coordinates by
        
    Returns:
        DataFrame with scaled coordinates
    """
    scaled_df = positions_df.copy()
    scaled_df['i'] = scaled_df['i'] * scale_factor
    scaled_df['j'] = scaled_df['j'] * scale_factor
    return scaled_df


def calculate_coordinate_overlap(positions_1: pd.DataFrame, positions_2: pd.DataFrame) -> float:
    """
    Calculate the overlap fraction between two coordinate datasets.
    
    Args:
        positions_1: First coordinate dataset
        positions_2: Second coordinate dataset
        
    Returns:
        Overlap fraction (0.0 to 1.0)
    """
    # Get coordinate ranges
    i1_min, i1_max = positions_1['i'].min(), positions_1['i'].max()
    j1_min, j1_max = positions_1['j'].min(), positions_1['j'].max()
    
    i2_min, i2_max = positions_2['i'].min(), positions_2['i'].max()
    j2_min, j2_max = positions_2['j'].min(), positions_2['j'].max()
    
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
    """
    Generate triangle hash for well-level data using the exact same approach as tile-by-tile.
    
    This replicates the proven find_triangles function but for stitched well data.
    """
    if len(positions_df) < 4:
        return pd.DataFrame()
    
    coords = positions_df[['i', 'j']].values
    
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
        df_result = df_combined.assign(magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5"))
        
        return df_result
        
    except Exception as e:
        print(f"Triangle hash generation failed: {e}")
        return pd.DataFrame()


def evaluate_well_match(
    vec_centers_0: pd.DataFrame, 
    vec_centers_1: pd.DataFrame, 
    threshold_triangle: float = 0.3, 
    threshold_point: float = 2.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Evaluate match using SINGLE fixed seed (42) for reproducible, fast results.
    """
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
                max_trials=1000
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
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample cells from the CENTER region for stable triangle hash alignment.
    """
    if len(cell_positions) <= max_cells:
        return cell_positions
    
    # Set fixed seed for reproducibility
    np.random.seed(random_state)
    
    print(f"=== Center Sampling (Seed: {random_state}) ===")
    print(f"Original cells: {len(cell_positions):,}")
    print(f"Target total: {max_cells:,}")
    print(f"Sampling CENTER {center_radius:.0%} region")
    
    # Calculate well boundaries and center
    i_coords = cell_positions['i'].values
    j_coords = cell_positions['j'].values
    
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
    df['dist_from_center_pixels'] = np.sqrt(
        (df['i'] - center_i) ** 2 + (df['j'] - center_j) ** 2
    )
    
    # Convert to normalized distance (0 = center, 1 = edge of well)
    df['dist_from_center_norm'] = df['dist_from_center_pixels'] / well_radius
    
    # Take cells from CENTER region
    center_mask = df['dist_from_center_norm'] <= center_radius
    center_cells = df[center_mask]
    
    print(f"Cells in center region (0%-{center_radius:.0%}): {len(center_cells):,}")
    
    if len(center_cells) == 0:
        print("❌ No cells found in center region! Using all cells")
        center_cells = df
    
    # Sample from center region
    if len(center_cells) > max_cells:
        center_sample = center_cells.nsmallest(max_cells, 'dist_from_center_norm')
    else:
        center_sample = center_cells
    
    # Remove temporary columns
    final_sample = center_sample.drop(columns=['dist_from_center_pixels', 'dist_from_center_norm'], errors='ignore')
    
    print(f"Final sampled: {len(final_sample):,} cells")
    
    return final_sample


def triangle_hash_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    max_cells_for_hash: int = 75000,
    threshold_triangle: float = 0.1,
    threshold_point: float = 2.0,
    min_score: float = 0.05,
    **kwargs
) -> pd.DataFrame:
    """
    Triangle hash alignment for well-level data.
    
    Args:
        phenotype_positions: Phenotype cell positions (should be pre-scaled)
        sbs_positions: SBS cell positions
        max_cells_for_hash: Maximum cells to use for triangle generation
        threshold_triangle: Triangle similarity threshold
        threshold_point: Point distance threshold
        min_score: Minimum score to accept alignment
        
    Returns:
        DataFrame with alignment parameters or empty DataFrame if failed
    """
    print(f"Triangle hash alignment with {len(phenotype_positions):,} phenotype and {len(sbs_positions):,} SBS cells")
    
    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()
    
    # Sample cells if datasets are too large
    if len(phenotype_positions) > max_cells_for_hash:
        pheno_subset = geographic_constrained_sampling(
            phenotype_positions,
            max_cells=max_cells_for_hash,
            center_radius=0.4,
            random_state=42
        )
    else:
        pheno_subset = phenotype_positions.copy()
        
    if len(sbs_positions) > max_cells_for_hash:
        sbs_subset = geographic_constrained_sampling(
            sbs_positions,
            max_cells=max_cells_for_hash,
            center_radius=0.4,
            random_state=42
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
        pheno_triangles, sbs_triangles, 
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point
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
        'rotation': rotation,
        'translation': translation,
        'score': score,
        'determinant': determinant,
        'transformation_type': 'triangle_hash_well_level',
        'triangles_matched': len(pheno_triangles),
        'approach': 'triangle_hash_after_scaling'
    }
    
    return pd.DataFrame([alignment])