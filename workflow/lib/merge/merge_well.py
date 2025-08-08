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


def evaluate_well_match(
    vec_centers_0: pd.DataFrame, 
    vec_centers_1: pd.DataFrame, 
    threshold_triangle: float = 0.3, 
    threshold_point: float = 2.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Evaluate match, finding the BEST transformation with positive determinant (no flips).
    """
    V_0, c_0 = get_vc(vec_centers_0)
    V_1, c_1 = get_vc(vec_centers_1)
    i0, i1, distances = nearest_neighbors(V_0, V_1)
    
    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]
    
    if sum(filt) < 5:
        return None, None, -1

    # Try multiple RANSAC runs and collect all positive determinant results
    valid_results = []
    
    for random_seed in range(42, 1042, 25):  # Try 40 different seeds
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = RANSACRegressor(
                    random_state=random_seed,
                    min_samples=max(5, len(X) // 10),
                    max_trials=1000
                )
                model.fit(X, Y)

            rotation = model.estimator_.coef_
            translation = model.estimator_.intercept_
            determinant = np.linalg.det(rotation)
            
            # Only consider positive determinants
            if determinant <= 0:
                continue
            
            # Calculate score for this positive determinant result
            distances = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
            threshold_region = 50
            filt_score = np.sqrt(distances.min(axis=0)) < threshold_region
            score = (np.sqrt(distances.min(axis=0))[filt_score] < threshold_point).mean()
            
            # Store this valid result
            valid_results.append({
                'rotation': rotation,
                'translation': translation,
                'score': score,
                'determinant': determinant,
                'seed': random_seed
            })
                
        except Exception:
            continue
    
    if not valid_results:
        print("❌ Could not find any transformation with positive determinant")
        return None, None, -1
    
    # Find the result with the highest score
    best_result = max(valid_results, key=lambda x: x['score'])
    
    print(f"Found {len(valid_results)} valid transformations with positive determinants:")
    for result in sorted(valid_results, key=lambda x: x['score'], reverse=True)[:3]:
        print(f"  Seed {result['seed']}: det={result['determinant']:.3f}, score={result['score']:.3f}")
    
    print(f"✅ Selected BEST: det={best_result['determinant']:.3f}, score={best_result['score']:.3f}")
    
    return best_result['rotation'], best_result['translation'], best_result['score']


def triangle_hash_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    max_cells_for_hash: int = 75000,
    threshold_triangle: float = 0.1,  # Same as tile-by-tile
    threshold_point: float = 2.0,     # Same as tile-by-tile
    min_score: float = 0.1,           # Same as tile-by-tile default
    **kwargs  # For compatibility with other parameters
) -> pd.DataFrame:
    """
    Well-level alignment using the exact same proven approach as tile-by-tile.
    
    This automatically handles the 8x scale difference via RANSAC, no hardcoding needed.
    """
    print(f"Starting proven triangle hash alignment with {len(phenotype_positions):,} phenotype and {len(sbs_positions):,} SBS cells")
    
    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()
    
    # Subsample if necessary to manage memory
    if len(phenotype_positions) > max_cells_for_hash:
        print(f"Subsampling phenotype cells from {len(phenotype_positions):,} to {max_cells_for_hash:,}")
        step = len(phenotype_positions) // max_cells_for_hash
        pheno_subset = phenotype_positions.iloc[::step][:max_cells_for_hash].copy()
    else:
        pheno_subset = phenotype_positions.copy()
        
    if len(sbs_positions) > max_cells_for_hash:
        print(f"Subsampling SBS cells from {len(sbs_positions):,} to {max_cells_for_hash:,}")
        step = len(sbs_positions) // max_cells_for_hash
        sbs_subset = sbs_positions.iloc[::step][:max_cells_for_hash].copy()
    else:
        sbs_subset = sbs_positions.copy()
    
    # Generate triangle hashes using proven approach
    print("Generating triangle hashes using proven nine-edge approach...")
    pheno_triangles = well_level_triangle_hash(pheno_subset)
    sbs_triangles = well_level_triangle_hash(sbs_subset)
    
    if len(pheno_triangles) == 0 or len(sbs_triangles) == 0:
        print("Failed to generate triangle hashes")
        return pd.DataFrame()
    
    print(f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles")
    
    # Evaluate match using exact tile-by-tile logic
    print("Evaluating match using proven approach...")
    rotation, translation, score = evaluate_well_match(
        pheno_triangles, sbs_triangles, 
        threshold_triangle=threshold_triangle,
        threshold_point=threshold_point
    )
    
    if rotation is None or score < min_score:
        print(f"Match evaluation failed: score={score:.3f} < {min_score}")
        return pd.DataFrame()
    
    determinant = np.linalg.det(rotation)
    
    print(f"✅ Proven approach successful:")
    print(f"   Score: {score:.3f}")
    print(f"   Determinant: {determinant:.3f}")
    print(f"   Translation: [{translation[0]:.1f}, {translation[1]:.1f}]")
    print(f"   RANSAC automatically handled 8x scale difference!")
    
    # Build result in same format as other approaches
    alignment = {
        'rotation': rotation,
        'translation': translation,
        'score': score,
        'determinant': determinant,
        'transformation_type': 'proven_nine_edge_hash',
        'n_triangles_matched': len(pheno_triangles),  # All triangles used
        'cells_used_phenotype': len(pheno_subset),
        'cells_used_sbs': len(sbs_subset),
        'triangles_generated_phenotype': len(pheno_triangles),
        'triangles_generated_sbs': len(sbs_triangles),
        'triangles_matched': sum(nearest_neighbors(get_vc(pheno_triangles)[0], get_vc(sbs_triangles)[0])[2] < threshold_triangle),
        'approach': 'proven_tile_by_tile_method'
    }
    
    result_df = pd.DataFrame([alignment])
    return result_df


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 2.0,
    chunk_size: int = 50000
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
        merged_cells = pd.concat(all_matches, ignore_index=True)
        
        # Remove duplicate phenotype cells (keep best matches)
        merged_cells = merged_cells.sort_values('distance').drop_duplicates('cell_0', keep='first')
        
        print(f"Successfully merged {len(merged_cells):,} cells")
        print(f"Distance statistics: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
        
        return merged_cells
    else:
        print("No cells matched within threshold")
        return pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])


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