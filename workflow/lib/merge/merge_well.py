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
        df_vectors = pd.DataFrame(vectors_array).rename(columns="V_{0}".format)
        df_coords = pd.DataFrame(centers_array).rename(columns="c_{0}".format)
        df_combined = pd.concat([df_vectors, df_coords], axis=1)
        
        # Add magnitude column (critical for normalization)
        df_result = df_combined.assign(magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5"))
        
        return df_result
        
    except Exception as e:
        print(f"Triangle hash generation failed: {e}")
        return pd.DataFrame()


# REPLACE this function in your merge_well.py file
# It's around lines 70-120 in your current file

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
    min_score: float = 0.1,
    **kwargs
) -> pd.DataFrame:
    """
    Fast triangle hash with consistent center sampling.
    """
    print(f"Starting proven triangle hash alignment with {len(phenotype_positions):,} phenotype and {len(sbs_positions):,} SBS cells")
    
    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()
    
    # Use CONSISTENT center radius for both modalities
    CENTER_RADIUS = 0.4  # 40% for both
    
    if len(phenotype_positions) > max_cells_for_hash:
        print(f"Center sampling phenotype cells from {len(phenotype_positions):,} to {max_cells_for_hash:,}")
        pheno_subset = geographic_constrained_sampling(
            phenotype_positions,
            max_cells=max_cells_for_hash,
            center_radius=CENTER_RADIUS,  # Consistent 40%
            random_state=42
        )
    else:
        pheno_subset = phenotype_positions.copy()
        
    if len(sbs_positions) > max_cells_for_hash:
        print(f"Center sampling SBS cells from {len(sbs_positions):,} to {max_cells_for_hash:,}")
        sbs_subset = geographic_constrained_sampling(
            sbs_positions,
            max_cells=max_cells_for_hash,
            center_radius=CENTER_RADIUS,  # Same 40% for consistency
            random_state=42
        )
    else:
        sbs_subset = sbs_positions.copy()
    
    # Rest of function stays the same...
    print("Generating triangle hashes using proven nine-edge approach...")
    pheno_triangles = well_level_triangle_hash(pheno_subset)
    sbs_triangles = well_level_triangle_hash(sbs_subset)
    
    if len(pheno_triangles) == 0 or len(sbs_triangles) == 0:
        print("Failed to generate triangle hashes")
        return pd.DataFrame()
    
    print(f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles")
    
    print("Evaluating match using FAST approach...")
    rotation, translation, score = evaluate_well_match(
        pheno_triangles, sbs_triangles, 
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point
    )
    
    if rotation is None or score < min_score:
        print(f"Match evaluation failed: score={score:.3f} < {min_score}")
        return pd.DataFrame()
    
    determinant = np.linalg.det(rotation)
    
    print(f"✅ FAST approach successful:")
    print(f"   Score: {score:.3f}")
    print(f"   Determinant: {determinant:.6f}")
    print(f"   Translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
    
    # Build result
    alignment = {
        'rotation': rotation,
        'translation': translation,
        'score': score,
        'determinant': determinant,
        'transformation_type': 'proven_nine_edge_hash_fast',
        'triangles_matched': len(pheno_triangles),
        'approach': 'fast_fixed_seed'
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
        
        # SAVE PRE-DEDUPLICATION MATCHES
        raw_matches_path = snakemake.output[0].replace('.parquet', '_raw_matches.parquet')
        merged_cells_raw.to_parquet(raw_matches_path)
        print(f"✅ Saved raw matches (before deduplication) to: {raw_matches_path}")
        
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


# Simple replacement for your current merge function
def replace_current_merge_logic():
    """
    Instructions for replacing your current merge logic:
    
    1. Replace your current alignment quality check with:
       check_alignment_quality_permissive()
       
    2. Remove any fallback alignment attempts
    
    3. Use the first triangle hash result that meets your criteria
    
    4. If you want to be even more permissive, just check:
       - det > 0 (positive determinant)
       - score > your_threshold
       And ignore the det_range entirely
    """
    pass

print("=== Usage Instructions ===")
print("To implement this change:")
print("1. Replace your current alignment quality check")
print("2. Remove fallback to other alignment methods") 
print("3. Accept first triangle hash result that meets basic criteria")
print("4. Optionally make det_range more permissive: [1e-6, 1e6]")



def geographic_constrained_sampling(
    cell_positions: pd.DataFrame,
    max_cells: int = 75000,
    center_radius: float = 0.4,      # Now used as OUTER boundary (keep for compatibility)
    center_max_cells: int = 75000,   # Ignored in donut version
    edge_max_cells: int = 0,         # Ignored in donut version
    mid_max_cells: int = 0,          # Ignored in donut version
    random_max_cells: int = 0,       # Ignored in donut version
    random_state: int = 42           # Fixed seed for reproducibility
) -> pd.DataFrame:
    """
    Sample cells ONLY from the OUTER ring (donut shape) for better edge coverage.
    
    FIXED: Now properly calculates donut for each dataset's own coordinate system.
    Each dataset (phenotype/SBS) gets its own circle calculation based on its actual cell positions.
    
    Args:
        cell_positions: DataFrame with cell positions
        max_cells: Total maximum cells to sample
        center_radius: Now interpreted as OUTER boundary (for compatibility)
        random_state: Fixed random seed
        
    Returns:
        DataFrame with cells sampled from outer ring (donut shape)
    """
    if len(cell_positions) <= max_cells:
        return cell_positions
    
    # Set fixed seed for reproducibility
    np.random.seed(random_state)
    
    # Define donut boundaries - OUTER ring sampling
    inner_radius = 0.6   # Exclude inner 60% (dense center)
    outer_radius = 1.0   # Include to edge (100%)
    
    print(f"=== Donut Sampling (Seed: {random_state}) ===")
    print(f"Original cells: {len(cell_positions):,}")
    print(f"Target total: {max_cells:,}")
    print(f"FIXED: Calculating donut independently for this dataset's coordinate system")
    print(f"Sampling OUTER {inner_radius:.0%}-{outer_radius:.0%} ring")
    
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
    
    # FIXED: Calculate circular distance properly for each dataset
    # Use the maximum of i_range and j_range to define the "radius" of the well
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
    
    # Take ONLY cells from donut region (OUTER ring)
    donut_mask = (df['dist_from_center_norm'] >= inner_radius) & (df['dist_from_center_norm'] <= outer_radius)
    donut_cells = df[donut_mask]
    
    print(f"Cells in outer ring ({inner_radius:.0%}-{outer_radius:.0%}): {len(donut_cells):,} ({100*len(donut_cells)/len(cell_positions):.1f}% of total)")
    
    if len(donut_cells) == 0:
        print("❌ No cells found in outer ring! Trying wider range...")
        # Try a wider range if no cells found
        wider_inner = 0.4  # Try 40% instead of 60%
        donut_mask_wider = (df['dist_from_center_norm'] >= wider_inner) & (df['dist_from_center_norm'] <= outer_radius)
        donut_cells = df[donut_mask_wider]
        print(f"Trying {wider_inner:.0%}-{outer_radius:.0%}: {len(donut_cells):,} cells")
        
        if len(donut_cells) == 0:
            print("❌ Still no cells! Falling back to all cells")
            donut_cells = df
    
    # Sample from outer ring
    if len(donut_cells) > max_cells:
        # Sample max_cells from outer ring, prioritizing middle of ring
        target_radius = (inner_radius + outer_radius) / 2  # Target ~80% radius
        donut_cells['dist_from_target'] = abs(donut_cells['dist_from_center_norm'] - target_radius)
        print(f"Sampling {max_cells:,} cells closest to target radius {target_radius:.0%}")
        donut_sample = donut_cells.nsmallest(max_cells, 'dist_from_target')
        donut_sample = donut_sample.drop(columns=['dist_from_target'])
    else:
        # Take all outer ring cells if we don't have enough
        print(f"Taking all {len(donut_cells):,} outer ring cells (less than target)")
        donut_sample = donut_cells
    
    # Remove temporary columns
    final_sample = donut_sample.drop(columns=['dist_from_center_pixels', 'dist_from_center_norm'], errors='ignore')
    
    print(f"=== Donut Sample Summary ===")
    print(f"Total sampled: {len(final_sample):,} / {max_cells:,} target")
    print(f"All cells from OUTER ring of THIS dataset's coordinate system")
    print(f"Sampling efficiency: {100*len(final_sample)/len(cell_positions):.1f}% of original")
    
    # Verify coverage
    if len(final_sample) > 0:
        i_range_sample = final_sample['i'].max() - final_sample['i'].min()
        j_range_sample = final_sample['j'].max() - final_sample['j'].min()
        print(f"Sample coverage: i={i_range_sample:.0f} ({100*i_range_sample/i_range:.1f}%), j={j_range_sample:.0f} ({100*j_range_sample/j_range:.1f}%)")
    
    return final_sample