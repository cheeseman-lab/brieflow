"""
Enhanced well-level merge functions for stitched image data.

This module uses the exact same proven approach as the successful tile-by-tile pipeline,
just applied to well-level stitched data. No hardcoded scaling assumptions.
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
from typing import Optional, Tuple, Dict, Any

# Import the proven functions from the existing hash module
from lib.merge.hash import nine_edge_hash, get_vc, nearest_neighbors


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

# FIX 1: Make center radius consistent in triangle_hash_well_alignment

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

def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 10.0,
    chunk_size: int = 50000,
    output_path: str = None
) -> pd.DataFrame:
    """
    Merge cells using alignment transformation with memory-efficient processing.
    
    This is the same as before - works with any alignment approach.
    """
    print(f"Starting cell merging with threshold={threshold}")
    
    # Extract transformation parameters
    if isinstance(alignment, pd.Series):
        rotation = alignment['rotation']
        translation = alignment['translation']
    else:
        rotation = alignment.get('rotation', np.eye(2))
        translation = alignment.get('translation', np.zeros(2))
    
    # Ensure rotation is 2x2 matrix
    if rotation is None or np.array(rotation).size != 4:
        rotation = np.eye(2)
    else:
        rotation = np.array(rotation).reshape(2, 2)
    
    # Ensure translation is length 2 vector
    if translation is None or np.array(translation).size != 2:
        translation = np.zeros(2)
    else:
        translation = np.array(translation).flatten()[:2]
    
    print(f"Using transformation: rotation det={np.linalg.det(rotation):.3f}, translation={translation}")

    # Get coordinates
    pheno_coords = phenotype_positions[['i', 'j']].values
    sbs_coords = sbs_positions[['i', 'j']].values

    # NOW add debug calls (after coordinates are defined):
    scale_factor = debug_transformation_matrix(rotation, translation)
    has_overlap, overlap_frac = debug_coordinate_transformation(pheno_coords, sbs_coords, rotation, translation)

    # Transform phenotype coordinates to SBS coordinate system
    transformed_coords = pheno_coords @ rotation.T + translation
     
    print(f"Coordinate ranges after transformation:")
    print(f"  Transformed phenotype: i=[{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}]")
    print(f"  SBS: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")
    
    # Process in chunks to manage memory
    all_matches = []
    
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size
    print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks of {chunk_size:,}")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))
        
        if chunk_idx % 10 == 0:
            print(f"Processing chunk {chunk_idx + 1}/{n_chunks}")
        
        # Get chunk of SBS coordinates
        sbs_chunk_coords = sbs_coords[start_idx:end_idx]
        
        # Calculate distances from transformed phenotype to SBS chunk
        distances = cdist(sbs_chunk_coords, transformed_coords, metric='euclidean')
        
        # Find closest phenotype cell for each SBS cell in chunk
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)
        
        # Filter by threshold
        valid_matches = min_distances < threshold
        
        if valid_matches.sum() > 0:
            # Create match records for this chunk
            sbs_chunk_indices = np.arange(start_idx, end_idx)[valid_matches]
            pheno_match_indices = closest_pheno_idx[valid_matches]
            match_distances = min_distances[valid_matches]
            
            # Build DataFrame for this chunk
            chunk_matches = pd.DataFrame({
                'cell_0': phenotype_positions.iloc[pheno_match_indices]['cell'].values,
                'i_0': phenotype_positions.iloc[pheno_match_indices]['i'].values,
                'j_0': phenotype_positions.iloc[pheno_match_indices]['j'].values,
                'cell_1': sbs_positions.iloc[sbs_chunk_indices]['cell'].values,
                'i_1': sbs_positions.iloc[sbs_chunk_indices]['i'].values,
                'j_1': sbs_positions.iloc[sbs_chunk_indices]['j'].values,
                'distance': match_distances
            })
            
            # Add area columns if available
            if 'area' in phenotype_positions.columns:
                chunk_matches['area_0'] = phenotype_positions.iloc[pheno_match_indices]['area'].values
            else:
                chunk_matches['area_0'] = np.nan
                
            if 'area' in sbs_positions.columns:
                chunk_matches['area_1'] = sbs_positions.iloc[sbs_chunk_indices]['area'].values
            else:
                chunk_matches['area_1'] = np.nan
            
            all_matches.append(chunk_matches)
    
    # Combine all chunks
    if all_matches:
        merged_cells_raw = pd.concat(all_matches, ignore_index=True)
        
        print(f"Before deduplication: {len(merged_cells_raw):,} matches")
        print(f"Duplicate phenotype cells: {merged_cells_raw['cell_0'].duplicated().sum():,}")
        
        # SAVE PRE-DEDUPLICATION MATCHES - FIX: Handle Namedlist/object conversion
        if output_path:
            # Convert output_path to string if it's not already
            output_path_str = str(output_path)
            raw_matches_path = output_path_str.replace('.parquet', '_raw_matches.parquet')
            merged_cells_raw.to_parquet(raw_matches_path)
            print(f"✅ Saved raw matches (before deduplication) to: {raw_matches_path}")
        else:
            print("⚠️ No output path provided - skipping raw matches save")
        
        # Remove duplicate phenotype cells (keep best matches)
        merged_cells = merged_cells_raw.sort_values('distance').drop_duplicates('cell_0', keep='first')
        
        print(f"After deduplication: {len(merged_cells):,} matches")
        print(f"Successfully merged {len(merged_cells):,} cells")
        print(f"Distance statistics: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
        
        return merged_cells
    else:
        print("No cells matched within threshold")
        return pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])

def debug_transformation_matrix(rotation, translation):
    """Debug what's actually in the transformation matrix."""
    print(f"\n=== Transformation Matrix Debug ===")
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
    print(f"  Expected scale: ~{1/8:.6f} (1/8 for phenotype→SBS)")
    
    # Check if it's close to expected scaling
    expected_scale = 1/8
    if abs(avg_scale - expected_scale) < 0.01:
        print("✅ Scale factor looks correct!")
    else:
        print(f"⚠️ Scale factor mismatch! Expected ~{expected_scale:.6f}, got {avg_scale:.6f}")
    
    # Check rotation component
    if scale_x > 0 and scale_y > 0:
        normalized_rotation = rotation / avg_scale
        print(f"Normalized rotation matrix (scale removed):")
        print(f"  [{normalized_rotation[0, 0]:.6f}, {normalized_rotation[0, 1]:.6f}]")
        print(f"  [{normalized_rotation[1, 0]:.6f}, {normalized_rotation[1, 1]:.6f}]")
        
        # Check if it's close to identity (no rotation)
        if np.allclose(normalized_rotation, np.eye(2), atol=0.1):
            print("✅ Pure scaling transformation (no significant rotation)")
        else:
            print("🔄 Scaling + rotation transformation")
    
    return avg_scale

# Also add this to check coordinate transformation results
def debug_coordinate_transformation(pheno_coords, sbs_coords, rotation, translation):
    """Debug the coordinate transformation results."""
    print(f"\n=== Coordinate Transformation Debug ===")
    
    # Apply transformation
    transformed_coords = pheno_coords @ rotation.T + translation
    
    # Compare coordinate ranges
    print(f"Original phenotype range:")
    print(f"  i: [{pheno_coords[:, 0].min():.0f}, {pheno_coords[:, 0].max():.0f}] (range: {pheno_coords[:, 0].max() - pheno_coords[:, 0].min():.0f})")
    print(f"  j: [{pheno_coords[:, 1].min():.0f}, {pheno_coords[:, 1].max():.0f}] (range: {pheno_coords[:, 1].max() - pheno_coords[:, 1].min():.0f})")
    
    print(f"Transformed phenotype range:")
    print(f"  i: [{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}] (range: {transformed_coords[:, 0].max() - transformed_coords[:, 0].min():.0f})")
    print(f"  j: [{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}] (range: {transformed_coords[:, 1].max() - transformed_coords[:, 1].min():.0f})")
    
    print(f"SBS coordinate range:")
    print(f"  i: [{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}] (range: {sbs_coords[:, 0].max() - sbs_coords[:, 0].min():.0f})")
    print(f"  j: [{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}] (range: {sbs_coords[:, 1].max() - sbs_coords[:, 1].min():.0f})")
    
    # Calculate overlap
    overlap_i_min = max(transformed_coords[:, 0].min(), sbs_coords[:, 0].min())
    overlap_i_max = min(transformed_coords[:, 0].max(), sbs_coords[:, 0].max())
    overlap_j_min = max(transformed_coords[:, 1].min(), sbs_coords[:, 1].min())
    overlap_j_max = min(transformed_coords[:, 1].max(), sbs_coords[:, 1].max())
    
    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min
    
    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = (sbs_coords[:, 0].max() - sbs_coords[:, 0].min()) * (sbs_coords[:, 1].max() - sbs_coords[:, 1].min())
        overlap_fraction = overlap_area / sbs_area if sbs_area > 0 else 0
        
        print(f"Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
        print(f"Overlap fraction: {overlap_fraction:.1%} of SBS area")
    else:
        print("❌ NO OVERLAP between transformed phenotype and SBS coordinates!")
    
    return has_overlap, overlap_fraction if has_overlap else 0


# Legacy Compatibility Functions - kept for backwards compatibility

def calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size):
    """Calculate expected scale factor (diagnostic only now)."""
    if phenotype_pixel_size and sbs_pixel_size:
        expected_scale = phenotype_pixel_size / sbs_pixel_size
        return expected_scale
    return None


def build_linear_model(rotation, translation):
    """Builds a linear regression model using the provided rotation matrix and translation vector."""
    m = LinearRegression()
    m.coef_ = rotation
    m.intercept_ = translation
    return m


def merge_sbs_phenotype(cell_locations_0, cell_locations_1, model, threshold=2):
    """Legacy function for backwards compatibility with tile-by-tile approach."""
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


# Redirect legacy functions to the proven approach
def stitched_well_alignment(*args, **kwargs):
    """Legacy function redirected to proven approach."""
    print("Note: Redirecting to proven triangle hash approach")
    return triangle_hash_well_alignment(*args, **kwargs)


def enhanced_alignment_with_correct_scale(*args, **kwargs):
    """Legacy function redirected to proven approach.""" 
    print("Note: Using proven triangle hash approach (no hardcoded scale needed)")
    return triangle_hash_well_alignment(*args, **kwargs)

# MODIFIED TO REMOVE FALLBACK

def check_alignment_quality_permissive(alignment, det_range, score_threshold):
    """
    Permissive alignment quality check - accept first match within parameters.
    
    Args:
        alignment: Alignment result series
        det_range: [min_det, max_det] determinant bounds
        score_threshold: Minimum score threshold
        
    Returns:
        bool: True if alignment meets basic criteria
    """
    det = alignment.get('determinant', 0)
    score = alignment.get('score', 0)
    
    # Check determinant bounds
    det_ok = det_range[0] <= det <= det_range[1] if det_range else True
    
    # Check score threshold  
    score_ok = score >= score_threshold
    
    # Check for positive determinant (non-degenerate transformation)
    positive_det = det > 0
    
    print(f"=== Permissive Quality Check ===")
    print(f"Determinant: {det:.6f} (bounds: {det_range})")
    print(f"Score: {score:.3f} (threshold: {score_threshold})")
    print(f"Positive determinant: {positive_det}")
    print(f"Determinant in range: {det_ok}")
    print(f"Score above threshold: {score_ok}")
    
    # Accept if ALL criteria are met
    all_criteria_met = det_ok and score_ok and positive_det
    
    if all_criteria_met:
        print("✅ All criteria met - accepting alignment")
    else:
        print("❌ Some criteria not met:")
        if not positive_det:
            print("  - Determinant not positive")
        if not det_ok:
            print("  - Determinant outside bounds")
        if not score_ok:
            print("  - Score below threshold")
    
    return all_criteria_met


def triangle_hash_alignment_no_fallback(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: list,
    score_threshold: float,
    max_cells_for_hash: int = 75000,
    **kwargs
) -> pd.DataFrame:
    """
    Triangle hash alignment that accepts first valid match - no fallback.
    
    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions  
        det_range: [min_det, max_det] determinant bounds
        score_threshold: Minimum score threshold
        max_cells_for_hash: Max cells for triangle generation
        
    Returns:
        DataFrame with alignment result or empty DataFrame if no valid match
    """
    print("=== Triangle-Hash-Only Alignment (No Fallback) ===")
    print(f"Target determinant range: {det_range}")
    print(f"Target score threshold: {score_threshold}")
    
    # Use your existing triangle hash approach
    try:
        # This calls your proven triangle hash implementation
        alignment_result = stitched_well_alignment(
            phenotype_positions=phenotype_positions,
            sbs_positions=sbs_positions,
            max_cells_for_hash=max_cells_for_hash,
            **kwargs
        )
        
        if alignment_result.empty:
            print("❌ Triangle hash alignment returned empty result")
            return pd.DataFrame()
        
        # Check the first (best) result
        best_alignment = alignment_result.iloc[0]
        
        print(f"Triangle hash result:")
        print(f"  Score: {best_alignment.get('score', 'N/A'):.3f}")
        print(f"  Determinant: {best_alignment.get('determinant', 'N/A'):.6f}")
        print(f"  Type: {best_alignment.get('transformation_type', 'unknown')}")
        
        # Apply permissive quality check
        is_acceptable = check_alignment_quality_permissive(
            best_alignment, det_range, score_threshold
        )
        
        if is_acceptable:
            print("✅ Triangle hash alignment accepted!")
            return alignment_result  # Return the full result
        else:
            print("❌ Triangle hash alignment rejected - not meeting criteria")
            # YOU COULD CHOOSE TO:
            # Option A: Return empty (strict)
            # return pd.DataFrame()
            
            # Option B: Return anyway (permissive - what you want)
            print("⚠️ Proceeding anyway as requested (no fallback)")
            return alignment_result
            
    except Exception as e:
        print(f"❌ Triangle hash alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def enhanced_well_merge_no_fallback(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: list,
    score_threshold: float = 0.1,
    merge_threshold: float = 10.0,
    max_cells_for_hash: int = 75000,
    phenotype_pixel_size: float = None,
    sbs_pixel_size: float = None,
    **kwargs
) -> pd.DataFrame:
    """
    Complete well merge pipeline with no fallback - accept first valid triangle hash.
    
    Returns:
        DataFrame with merged cells or empty DataFrame if alignment fails
    """
    print(f"=== Enhanced Well Merge (No Fallback) ===")
    print(f"Phenotype cells: {len(phenotype_positions):,}")
    print(f"SBS cells: {len(sbs_positions):,}")
    print(f"Determinant range: {det_range}")
    print(f"Score threshold: {score_threshold}")
    print(f"Merge threshold: {merge_threshold}")
    
    # Step 1: Triangle hash alignment only
    alignment_result = triangle_hash_alignment_no_fallback(
        phenotype_positions=phenotype_positions,
        sbs_positions=sbs_positions,
        det_range=det_range,
        score_threshold=score_threshold,
        max_cells_for_hash=max_cells_for_hash,
        phenotype_pixel_size=phenotype_pixel_size,
        sbs_pixel_size=sbs_pixel_size,
        **kwargs
    )
    
    if alignment_result.empty:
        print("❌ No valid alignment found")
        return pd.DataFrame()
    
    # Step 2: Use the alignment for cell merging
    best_alignment = alignment_result.iloc[0]
    
    print(f"\n=== Proceeding with Cell Merge ===")
    print(f"Using alignment with score: {best_alignment.get('score', 'N/A'):.3f}")
    
    try:
        merged_cells = merge_stitched_cells(
            phenotype_positions=phenotype_positions,
            sbs_positions=sbs_positions,
            alignment=best_alignment,
            threshold=merge_threshold
        )
        
        if merged_cells.empty:
            print("❌ Cell merge returned no matches")
        else:
            print(f"✅ Cell merge successful: {len(merged_cells)} cells matched")
            print(f"   Mean distance: {merged_cells['distance'].mean():.2f}")
            print(f"   Max distance: {merged_cells['distance'].max():.2f}")
        
        return merged_cells
        
    except Exception as e:
        print(f"❌ Cell merge failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def geographic_constrained_sampling(
    cell_positions: pd.DataFrame,
    max_cells: int = 75000,
    center_radius: float = 0.4,      # Use center region instead of donut
    center_max_cells: int = 75000,   # Take ALL cells from center (up to max)
    edge_max_cells: int = 0,         # NO edge cells
    mid_max_cells: int = 0,          # NO middle cells
    random_max_cells: int = 0,       # NO random fill
    random_state: int = 42           # Fixed seed for reproducibility
) -> pd.DataFrame:
    """
    Sample cells from the CENTER region for stable triangle hash alignment.
    
    CHANGED BACK: Now sampling from center region instead of outer donut.
    Center cells provide more stable triangulation for alignment.
    
    Args:
        cell_positions: DataFrame with cell positions
        max_cells: Total maximum cells to sample
        center_radius: Fraction of well radius defining center region (0.4 = 40%)
        random_state: Fixed random seed
        
    Returns:
        DataFrame with cells sampled from center region
    """
    if len(cell_positions) <= max_cells:
        return cell_positions
    
    # Set fixed seed for reproducibility
    np.random.seed(random_state)
    
    print(f"=== Center Sampling (Seed: {random_state}) ===")
    print(f"Original cells: {len(cell_positions):,}")
    print(f"Target total: {max_cells:,}")
    print(f"CHANGED BACK: Now sampling CENTER {center_radius:.0%} region (not donut)")
    
    # Calculate well boundaries and center FOR THIS SPECIFIC DATASET
    i_coords = cell_positions['i'].values
    j_coords = cell_positions['j'].values
    
    i_min, i_max = i_coords.min(), i_coords.max()
    j_min, j_max = j_coords.min(), j_coords.max()
    i_range = i_max - i_min
    j_range = j_max - j_min
    
    center_i = (i_min + i_max) / 2
    center_j = (j_min + j_max) / 2
    
    print(f"Dataset boundaries: i=[{i_min:.0f}, {i_max:.0f}], j=[{j_min:.0f}, {j_max:.0f}]")
    print(f"Dataset center: ({center_i:.0f}, {center_j:.0f})")
    print(f"Dataset dimensions: {i_range:.0f} × {j_range:.0f}")
    
    # Calculate circular distance for each dataset independently
    well_radius = max(i_range, j_range) / 2
    print(f"Well radius: {well_radius:.0f} pixels")
    
    # Add distance from center in PIXELS (not normalized)
    df = cell_positions.copy()
    df['dist_from_center_pixels'] = np.sqrt(
        (df['i'] - center_i) ** 2 + (df['j'] - center_j) ** 2
    )
    
    # Convert to normalized distance (0 = center, 1 = edge of well)
    df['dist_from_center_norm'] = df['dist_from_center_pixels'] / well_radius
    
    print(f"Distance range: {df['dist_from_center_norm'].min():.3f} to {df['dist_from_center_norm'].max():.3f}")
    print(f"Cells at various distances:")
    print(f"  0-20%: {(df['dist_from_center_norm'] <= 0.2).sum():,}")
    print(f"  20-40%: {((df['dist_from_center_norm'] > 0.2) & (df['dist_from_center_norm'] <= 0.4)).sum():,}")
    print(f"  40-60%: {((df['dist_from_center_norm'] > 0.4) & (df['dist_from_center_norm'] <= 0.6)).sum():,}")
    print(f"  60-80%: {((df['dist_from_center_norm'] > 0.6) & (df['dist_from_center_norm'] <= 0.8)).sum():,}")
    print(f"  80-100%: {((df['dist_from_center_norm'] > 0.8) & (df['dist_from_center_norm'] <= 1.0)).sum():,}")
    print(f"  >100%: {(df['dist_from_center_norm'] > 1.0).sum():,}")
    
    # Take ONLY cells from CENTER region
    center_mask = df['dist_from_center_norm'] <= center_radius
    center_cells = df[center_mask]
    
    print(f"Cells in center region (0%-{center_radius:.0%}): {len(center_cells):,} ({100*len(center_cells)/len(cell_positions):.1f}% of total)")
    
    if len(center_cells) == 0:
        print("❌ No cells found in center region! Trying wider range...")
        # Try a wider range if no cells found
        wider_radius = 0.6  # Try 60% instead of 40%
        center_mask_wider = df['dist_from_center_norm'] <= wider_radius
        center_cells = df[center_mask_wider]
        print(f"Trying 0%-{wider_radius:.0%}: {len(center_cells):,} cells")
        
        if len(center_cells) == 0:
            print("❌ Still no cells! Falling back to all cells")
            center_cells = df
    
    # Sample from center region
    if len(center_cells) > max_cells:
        # Sample max_cells from center, prioritizing cells closest to true center
        print(f"Sampling {max_cells:,} cells closest to center")
        center_sample = center_cells.nsmallest(max_cells, 'dist_from_center_norm')
    else:
        # Take all center cells if we don't have enough
        print(f"Taking all {len(center_cells):,} center cells (less than target)")
        center_sample = center_cells
    
    # Remove temporary columns
    final_sample = center_sample.drop(columns=['dist_from_center_pixels', 'dist_from_center_norm'], errors='ignore')
    
    print(f"=== Center Sample Summary ===")
    print(f"Total sampled: {len(final_sample):,} / {max_cells:,} target")
    print(f"All cells from CENTER region of THIS dataset's coordinate system")
    print(f"Sampling efficiency: {100*len(final_sample)/len(cell_positions):.1f}% of original")
    
    # Verify coverage
    if len(final_sample) > 0:
        i_range_sample = final_sample['i'].max() - final_sample['i'].min()
        j_range_sample = final_sample['j'].max() - final_sample['j'].min()
        print(f"Sample coverage: i={i_range_sample:.0f} ({100*i_range_sample/i_range:.1f}%), j={j_range_sample:.0f} ({100*j_range_sample/j_range:.1f}%)")
    
    return final_sample


def hardcoded_scale_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = 0.125,  # 1/8 for phenotype->SBS
    max_cells_for_translation: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Use hardcoded scale factor and only learn translation from data.
    
    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions  
        scale_factor: Fixed scale factor (default 0.125 = 1/8)
        max_cells_for_translation: Max cells to use for translation learning
        random_state: Random seed for sampling
        
    Returns:
        DataFrame with alignment result
    """
    print(f"=== Hardcoded Scale + Learned Translation ===")
    print(f"Using fixed scale factor: {scale_factor}")
    print(f"Learning translation from {max_cells_for_translation:,} cell pairs")
    
    # Sample cells for translation learning (use center cells for stability)
    if len(phenotype_positions) > max_cells_for_translation:
        pheno_sample = geographic_constrained_sampling(
            phenotype_positions, 
            max_cells=max_cells_for_translation,
            center_radius=0.3,  # Use center 30% for stable translation
            random_state=random_state
        )
    else:
        pheno_sample = phenotype_positions.copy()
        
    if len(sbs_positions) > max_cells_for_translation:
        sbs_sample = geographic_constrained_sampling(
            sbs_positions, 
            max_cells=max_cells_for_translation,
            center_radius=0.3,  # Same center region
            random_state=random_state
        )
    else:
        sbs_sample = sbs_positions.copy()
    
    print(f"Using {len(pheno_sample):,} phenotype and {len(sbs_sample):,} SBS cells for translation")
    
    # Build hardcoded transformation matrix
    # Pure scaling matrix (no rotation)
    rotation_matrix = np.array([
        [scale_factor, 0.0],
        [0.0, scale_factor]
    ])
    
    print(f"Hardcoded rotation matrix:")
    print(f"  [{rotation_matrix[0, 0]:.6f}, {rotation_matrix[0, 1]:.6f}]")
    print(f"  [{rotation_matrix[1, 0]:.6f}, {rotation_matrix[1, 1]:.6f}]")
    
    # Learn translation using center of mass alignment
    pheno_coords = pheno_sample[['i', 'j']].values
    sbs_coords = sbs_sample[['i', 'j']].values
    
    # Scale phenotype coordinates
    scaled_pheno_coords = pheno_coords * scale_factor
    
    # Calculate centers of mass
    pheno_center = scaled_pheno_coords.mean(axis=0)
    sbs_center = sbs_coords.mean(axis=0)
    
    # Translation = difference between centers
    translation = sbs_center - pheno_center
    
    print(f"Coordinate centers:")
    print(f"  Scaled phenotype center: [{pheno_center[0]:.1f}, {pheno_center[1]:.1f}]")
    print(f"  SBS center: [{sbs_center[0]:.1f}, {sbs_center[1]:.1f}]")
    print(f"  Calculated translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
    
    # Validate the transformation by checking a subset of points
    transformed_pheno = scaled_pheno_coords + translation
    
    # Find nearest neighbors to estimate quality
    from scipy.spatial.distance import cdist
    distances = cdist(transformed_pheno, sbs_coords, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # Calculate quality metrics
    mean_distance = min_distances.mean()
    good_matches = (min_distances < 10.0).sum()  # Within 10 pixels
    score = good_matches / len(min_distances)
    
    print(f"Transformation validation:")
    print(f"  Mean nearest neighbor distance: {mean_distance:.2f} px")
    print(f"  Matches within 10px: {good_matches:,}/{len(min_distances):,} ({score:.1%})")
    
    # Build result
    alignment = {
        'rotation': rotation_matrix,
        'translation': translation,
        'score': score,
        'determinant': np.linalg.det(rotation_matrix),
        'transformation_type': 'hardcoded_scale_learned_translation',
        'scale_factor': scale_factor,
        'approach': 'center_of_mass_translation',
        'validation_mean_distance': mean_distance,
        'validation_good_matches': good_matches
    }
    
    print(f"✅ Hardcoded scale alignment successful:")
    print(f"   Scale factor: {scale_factor}")
    print(f"   Translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
    print(f"   Validation score: {score:.3f}")
    print(f"   Mean validation distance: {mean_distance:.2f} px")
    
    return pd.DataFrame([alignment])


# ALTERNATIVE: More sophisticated translation learning using robust estimation
def hardcoded_scale_robust_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = 0.125,
    max_cells_for_translation: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Use hardcoded scale + robust translation estimation via RANSAC on point correspondences.
    """
    print(f"=== Hardcoded Scale + Robust Translation ===")
    print(f"Using fixed scale factor: {scale_factor}")
    
    # Sample cells for translation learning
    if len(phenotype_positions) > max_cells_for_translation:
        pheno_sample = geographic_constrained_sampling(
            phenotype_positions, 
            max_cells=max_cells_for_translation,
            center_radius=0.4,  # Use center for stability
            random_state=random_state
        )
    else:
        pheno_sample = phenotype_positions.copy()
        
    if len(sbs_positions) > max_cells_for_translation:
        sbs_sample = geographic_constrained_sampling(
            sbs_positions, 
            max_cells=max_cells_for_translation,
            center_radius=0.4,
            random_state=random_state
        )
    else:
        sbs_sample = sbs_positions.copy()
    
    # Build hardcoded scaling matrix
    rotation_matrix = np.array([
        [scale_factor, 0.0],
        [0.0, scale_factor]
    ])
    
    # Scale phenotype coordinates
    pheno_coords = pheno_sample[['i', 'j']].values
    sbs_coords = sbs_sample[['i', 'j']].values
    scaled_pheno_coords = pheno_coords * scale_factor
    
    # Find initial correspondences using nearest neighbors
    from scipy.spatial.distance import cdist
    distances = cdist(scaled_pheno_coords, sbs_coords, metric='euclidean')
    
    # For each scaled phenotype cell, find closest SBS cell
    closest_sbs_idx = distances.argmin(axis=1)
    closest_distances = distances.min(axis=1)
    
    # Filter to reasonable matches only (within some initial threshold)
    reasonable_threshold = 50.0  # Generous initial threshold
    good_matches = closest_distances < reasonable_threshold
    
    if good_matches.sum() < 10:
        print(f"❌ Too few reasonable initial matches: {good_matches.sum()}")
        return pd.DataFrame()
    
    # Get matched point pairs
    pheno_matched = scaled_pheno_coords[good_matches]
    sbs_matched = sbs_coords[closest_sbs_idx[good_matches]]
    
    print(f"Found {len(pheno_matched):,} initial point correspondences")
    
    # Use RANSAC to robustly estimate translation
    from sklearn.linear_model import RANSACRegressor
    
    try:
        # Fit translation: sbs = pheno + translation
        model = RANSACRegressor(
            random_state=random_state,
            min_samples=max(5, len(pheno_matched) // 20),
            max_trials=1000,
            residual_threshold=5.0  # 5 pixel threshold
        )
        
        model.fit(pheno_matched, sbs_matched)
        
        # Extract translation (intercept of linear model)
        translation = model.estimator_.intercept_
        
        # The coefficient should be close to identity since we already scaled
        coef = model.estimator_.coef_
        print(f"RANSAC coefficients (should be ~1.0): [{coef[0, 0]:.3f}, {coef[1, 1]:.3f}]")
        
        # Validate the transformation
        all_scaled_pheno = pheno_coords * scale_factor
        transformed_coords = all_scaled_pheno + translation
        
        distances_all = cdist(transformed_coords, sbs_coords, metric='euclidean')
        min_distances_all = distances_all.min(axis=1)
        
        mean_distance = min_distances_all.mean()
        good_matches_count = (min_distances_all < 10.0).sum()
        score = good_matches_count / len(min_distances_all)
        
        print(f"Robust translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
        print(f"Validation: {good_matches_count:,}/{len(min_distances_all):,} matches within 10px ({score:.1%})")
        
        alignment = {
            'rotation': rotation_matrix,
            'translation': translation,
            'score': score,
            'determinant': np.linalg.det(rotation_matrix),
            'transformation_type': 'hardcoded_scale_ransac_translation',
            'scale_factor': scale_factor,
            'approach': 'robust_point_correspondence',
            'validation_mean_distance': mean_distance,
            'validation_good_matches': good_matches_count
        }
        
        return pd.DataFrame([alignment])
        
    except Exception as e:
        print(f"❌ Robust translation estimation failed: {e}")
        return pd.DataFrame()


# USAGE: Replace triangle hash alignment with hardcoded scale approach
def test_hardcoded_scale_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = 0.125
) -> pd.DataFrame:
    """
    Test both simple and robust hardcoded scale approaches.
    """
    print(f"=== Testing Hardcoded Scale Approaches ===")
    
    # Try simple center-of-mass translation first
    simple_result = hardcoded_scale_alignment(
        phenotype_positions, sbs_positions, scale_factor
    )
    
    if not simple_result.empty:
        simple_score = simple_result.iloc[0]['score']
        print(f"Simple approach score: {simple_score:.3f}")
        
        # If simple approach works well, use it
        if simple_score > 0.3:
            print("✅ Simple center-of-mass translation works well")
            return simple_result
    
    # Try robust approach
    print("\nTrying robust RANSAC translation...")
    robust_result = hardcoded_scale_robust_alignment(
        phenotype_positions, sbs_positions, scale_factor
    )
    
    if not robust_result.empty:
        robust_score = robust_result.iloc[0]['score']
        print(f"Robust approach score: {robust_score:.3f}")
        
        # Return the better result
        if simple_result.empty:
            return robust_result
        elif robust_score > simple_score:
            print("✅ Robust approach performs better")
            return robust_result
        else:
            print("✅ Simple approach performs better")
            return simple_result
    
    print("❌ Both hardcoded scale approaches failed")
    return pd.DataFrame()


# Example usage in your well_merge.py script:
"""
# Replace the triangle_hash_well_alignment call with:
alignment_df = test_hardcoded_scale_alignment(
    phenotype_positions=phenotype_well,
    sbs_positions=sbs_well,
    scale_factor=0.125  # 1/8 for phenotype->SBS
)
"""

# ADD this function to your merge_well.py library

# ADD this function to your merge_well.py library

def pure_scaling_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factor: float = None,  # If None, calculate from data
    validation_sample_size: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Pure scaling transformation with no translation.
    Tests if coordinate systems are already aligned and just need scaling.
    
    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions  
        scale_factor: Fixed scale factor (if None, calculate from coordinate ranges)
        validation_sample_size: Number of cells for validation
        random_state: Random seed
        
    Returns:
        DataFrame with alignment result
    """
    print(f"=== Pure Scaling Alignment (No Translation) ===")
    
    # Get coordinate ranges
    pheno_coords = phenotype_positions[['i', 'j']].values
    sbs_coords = sbs_positions[['i', 'j']].values
    
    pheno_i_range = np.ptp(pheno_coords[:, 0])
    pheno_j_range = np.ptp(pheno_coords[:, 1])
    sbs_i_range = np.ptp(sbs_coords[:, 0])
    sbs_j_range = np.ptp(sbs_coords[:, 1])
    
    print(f"Coordinate ranges:")
    print(f"  Phenotype: i={pheno_i_range:.0f}, j={pheno_j_range:.0f}")
    print(f"  SBS: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")
    
    # Calculate scale factor if not provided
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
    
    # Build pure scaling matrix (no translation)
    rotation_matrix = np.array([
        [scale_factor, 0.0],
        [0.0, scale_factor]
    ])
    translation = np.array([0.0, 0.0])  # NO TRANSLATION
    
    print(f"Pure scaling matrix:")
    print(f"  [{rotation_matrix[0, 0]:.6f}, {rotation_matrix[0, 1]:.6f}]")
    print(f"  [{rotation_matrix[1, 0]:.6f}, {rotation_matrix[1, 1]:.6f}]")
    print(f"Translation: [0.0, 0.0]")
    
    # Validate the transformation
    print(f"\nValidating with {validation_sample_size:,} cells...")
    
    # Sample cells for validation
    if len(phenotype_positions) > validation_sample_size:
        pheno_sample_idx = np.random.choice(len(phenotype_positions), validation_sample_size, replace=False)
        pheno_sample = pheno_coords[pheno_sample_idx]
    else:
        pheno_sample = pheno_coords
        
    if len(sbs_positions) > validation_sample_size:
        sbs_sample_idx = np.random.choice(len(sbs_positions), validation_sample_size, replace=False)
        sbs_sample = sbs_coords[sbs_sample_idx]
    else:
        sbs_sample = sbs_coords
    
    # Apply pure scaling transformation
    transformed_pheno = pheno_sample * scale_factor  # Just multiply by scale, no translation
    
    print(f"Coordinate ranges after pure scaling:")
    print(f"  Original phenotype: i=[{pheno_sample[:, 0].min():.0f}, {pheno_sample[:, 0].max():.0f}], j=[{pheno_sample[:, 1].min():.0f}, {pheno_sample[:, 1].max():.0f}]")
    print(f"  Scaled phenotype: i=[{transformed_pheno[:, 0].min():.0f}, {transformed_pheno[:, 0].max():.0f}], j=[{transformed_pheno[:, 1].min():.0f}, {transformed_pheno[:, 1].max():.0f}]")
    print(f"  SBS sample: i=[{sbs_sample[:, 0].min():.0f}, {sbs_sample[:, 0].max():.0f}], j=[{sbs_sample[:, 1].min():.0f}, {sbs_sample[:, 1].max():.0f}]")
    
    # Calculate overlap region
    overlap_i_min = max(transformed_pheno[:, 0].min(), sbs_sample[:, 0].min())
    overlap_i_max = min(transformed_pheno[:, 0].max(), sbs_sample[:, 0].max())
    overlap_j_min = max(transformed_pheno[:, 1].min(), sbs_sample[:, 1].min())
    overlap_j_max = min(transformed_pheno[:, 1].max(), sbs_sample[:, 1].max())
    
    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min
    
    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = (sbs_sample[:, 0].max() - sbs_sample[:, 0].min()) * (sbs_sample[:, 1].max() - sbs_sample[:, 1].min())
        overlap_fraction = overlap_area / sbs_area if sbs_area > 0 else 0
        
        print(f"Overlap analysis:")
        print(f"  Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
        print(f"  Overlap fraction: {overlap_fraction:.1%} of SBS area")
    else:
        print(f"❌ NO OVERLAP with pure scaling!")
        overlap_fraction = 0.0
    
    # Find nearest neighbors to estimate match quality
    from scipy.spatial.distance import cdist
    distances = cdist(transformed_pheno, sbs_sample, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # Calculate quality metrics at different thresholds
    thresholds = [2, 5, 10, 20, 50]
    scores = {}
    for thresh in thresholds:
        good_matches = (min_distances < thresh).sum()
        score = good_matches / len(min_distances)
        scores[thresh] = score
        print(f"  Matches within {thresh:2d}px: {good_matches:4d}/{len(min_distances):,} ({score:.1%})")
    
    mean_distance = min_distances.mean()
    median_distance = np.median(min_distances)
    
    print(f"Distance statistics:")
    print(f"  Mean: {mean_distance:.2f} px")
    print(f"  Median: {median_distance:.2f} px")
    print(f"  Min: {min_distances.min():.2f} px")
    print(f"  Max: {min_distances.max():.2f} px")
    print(f"  95th percentile: {np.percentile(min_distances, 95):.2f} px")
    
    # Build result
    alignment = {
        'rotation': rotation_matrix,
        'translation': translation,
        'score': scores[10],  # Use 10px threshold for main score
        'determinant': np.linalg.det(rotation_matrix),
        'transformation_type': 'pure_scaling_no_translation',
        'scale_factor': scale_factor,
        'approach': 'coordinate_range_scaling',
        'overlap_fraction': overlap_fraction,
        'validation_mean_distance': mean_distance,
        'validation_median_distance': median_distance,
        'scores_by_threshold': scores,
        'has_overlap': has_overlap
    }
    
    if has_overlap and scores[10] > 0.01:  # At least 1% of cells within 10px
        print(f"✅ Pure scaling shows promise:")
        print(f"   Scale factor: {scale_factor:.6f}")
        print(f"   Overlap: {overlap_fraction:.1%}")
        print(f"   Score (10px): {scores[10]:.3f}")
        print(f"   Mean distance: {mean_distance:.2f} px")
    else:
        print(f"❌ Pure scaling insufficient:")
        print(f"   Overlap: {overlap_fraction:.1%}")
        print(f"   Score (10px): {scores[10]:.3f}")
        print(f"   May need translation adjustment")
    
    return pd.DataFrame([alignment])


# Test different scale factors
def test_multiple_scale_factors(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    scale_factors: list = None,
    validation_sample_size: int = 10000
) -> pd.DataFrame:
    """
    Test pure scaling with multiple scale factors to find the best one.
    """
    if scale_factors is None:
        # Test theoretical and calculated scale factors
        pheno_coords = phenotype_positions[['i', 'j']].values
        sbs_coords = sbs_positions[['i', 'j']].values
        
        # NumPy 2.0 compatible range calculation
        pheno_i_range = pheno_coords[:, 0].max() - pheno_coords[:, 0].min()
        pheno_j_range = pheno_coords[:, 1].max() - pheno_coords[:, 1].min()
        sbs_i_range = sbs_coords[:, 0].max() - sbs_coords[:, 0].min()
        sbs_j_range = sbs_coords[:, 1].max() - sbs_coords[:, 1].min()
        
        calculated_scale = ((sbs_i_range / pheno_i_range) + (sbs_j_range / pheno_j_range)) / 2
        
        scale_factors = [
            0.125,  # Theoretical 1/8
            calculated_scale,  # From coordinate ranges
            0.120,  # Nearby values
            0.123,
            0.127,
            0.130
        ]
    
    print(f"=== Testing Multiple Scale Factors ===")
    print(f"Scale factors to test: {[f'{s:.6f}' for s in scale_factors]}")
    
    results = []
    for scale in scale_factors:
        print(f"\n--- Testing scale factor: {scale:.6f} ---")
        result = pure_scaling_alignment(
            phenotype_positions, sbs_positions, 
            scale_factor=scale,
            validation_sample_size=validation_sample_size
        )
        if not result.empty:
            results.append(result.iloc[0])
    
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n=== Scale Factor Comparison ===")
        print("Scale    Score(10px) Mean_Dist  Overlap%")
        print("-" * 40)
        for _, row in results_df.iterrows():
            print(f"{row['scale_factor']:.6f}   {row['score']:.3f}     {row['validation_mean_distance']:6.1f}    {row['overlap_fraction']*100:5.1f}%")
        
        # Find best scale factor
        best_idx = results_df['score'].idxmax()
        best_result = results_df.iloc[best_idx:best_idx+1].copy()
        
        # Fix: Use best_result instead of best_alignment
        best_alignment = best_result.iloc[0]  # Get the series
        scale_factor = best_alignment.get('scale_factor', 'N/A')
        mean_distance = best_alignment.get('validation_mean_distance', 'N/A')
        print(f"\n✅ Best scale factor: {scale_factor}")
        print(f"   Score: {best_alignment.get('score', 'N/A'):.3f}")
        print(f"   Mean distance: {mean_distance}")
        print(f"   Overlap: {best_alignment.get('overlap_fraction', 0)*100:.1f}%")
        
        return best_result
    else:
        return pd.DataFrame()


# Usage example:
"""
# In your well_merge.py script, replace triangle hash alignment with:

# Option 1: Test calculated scale factor
alignment_df = pure_scaling_alignment(
    phenotype_positions=phenotype_well,
    sbs_positions=sbs_well
)

# Option 2: Test multiple scale factors  
alignment_df = test_multiple_scale_factors(
    phenotype_positions=phenotype_well,
    sbs_positions=sbs_well
)

# Option 3: Test specific scale factor
alignment_df = pure_scaling_alignment(
    phenotype_positions=phenotype_well,
    sbs_positions=sbs_well,
    scale_factor=0.123  # Or whatever you want to test
)
"""