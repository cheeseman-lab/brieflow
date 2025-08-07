"""
Memory-efficient well-level merge functions using stitched cell positions.
Only supports memory-efficient triangle hashing for large cell datasets.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
import gc
from typing import Tuple

# Import existing functions
from lib.merge.hash import get_vectors, get_vc


def subsample_cells_for_alignment(
    cell_positions: pd.DataFrame, 
    max_cells: int = 50000,
    spatial_bins: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Subsample cells while preserving spatial distribution.
    
    Args:
        cell_positions: DataFrame with cell positions
        max_cells: Maximum number of cells to keep
        spatial_bins: Number of spatial bins per dimension for stratified sampling
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled DataFrame
    """
    if len(cell_positions) <= max_cells:
        return cell_positions
    
    print(f"Subsampling from {len(cell_positions)} to {max_cells} cells")
    
    # Create spatial bins
    i_bins = pd.cut(cell_positions['i'], bins=spatial_bins, labels=False)
    j_bins = pd.cut(cell_positions['j'], bins=spatial_bins, labels=False)
    cell_positions = cell_positions.copy()
    cell_positions['spatial_bin'] = i_bins * spatial_bins + j_bins
    
    # Stratified sampling within each spatial bin
    np.random.seed(random_state)
    
    subsampled_parts = []
    cells_per_bin = max_cells // (spatial_bins * spatial_bins)
    
    for bin_id in range(spatial_bins * spatial_bins):
        bin_cells = cell_positions[cell_positions['spatial_bin'] == bin_id]
        if len(bin_cells) > 0:
            n_sample = min(len(bin_cells), max(cells_per_bin, 1))
            subsampled = bin_cells.sample(n=n_sample, random_state=random_state)
            subsampled_parts.append(subsampled)
    
    result = pd.concat(subsampled_parts, ignore_index=True)
    result = result.drop('spatial_bin', axis=1)
    
    print(f"Subsampled to {len(result)} cells")
    return result


def hash_cell_positions_memory_efficient(
    cell_positions: pd.DataFrame,
    max_cells_for_hash: int = 50000
) -> pd.DataFrame:
    """
    Generate triangle hash with memory efficiency.
    
    Args:
        cell_positions: DataFrame with cell positions
        max_cells_for_hash: Maximum cells to use for hashing
        
    Returns:
        DataFrame with triangle hash features
    """
    if len(cell_positions) < 4:
        print(f"Warning: Only {len(cell_positions)} cells found, need at least 4")
        return pd.DataFrame()
    
    # Subsample if too many cells
    if len(cell_positions) > max_cells_for_hash:
        cell_positions_sampled = subsample_cells_for_alignment(
            cell_positions, max_cells_for_hash
        )
    else:
        cell_positions_sampled = cell_positions
    
    # Extract coordinates and compute Delaunay triangulation
    coordinates = cell_positions_sampled[['i', 'j']].values
    
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
        df_result['well'] = cell_positions['well'].iloc[0]
        
        print(f"Generated {len(df_result)} triangles from {len(cell_positions_sampled)} cells")
        return df_result
        
    except Exception as e:
        print(f"Error computing triangulation: {e}")
        return pd.DataFrame()


def chunked_nearest_neighbors(
    V_0: np.ndarray, 
    V_1: np.ndarray, 
    chunk_size: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient nearest neighbor search using chunks.
    
    Args:
        V_0: First set of vectors
        V_1: Second set of vectors  
        chunk_size: Size of chunks to process
        
    Returns:
        Tuple of (indices_0, indices_1, distances)
    """
    n_0 = V_0.shape[0]
    
    all_ix_0 = []
    all_ix_1 = []
    all_distances = []
    
    print(f"Processing {n_0} vectors in chunks of {chunk_size}")
    
    for start_idx in range(0, n_0, chunk_size):
        end_idx = min(start_idx + chunk_size, n_0)
        chunk_V_0 = V_0[start_idx:end_idx]
        
        # Compute distances for this chunk
        chunk_distances = cdist(chunk_V_0, V_1, metric='sqeuclidean')
        chunk_distances = np.sqrt(chunk_distances)
        
        # Find nearest neighbors for this chunk
        chunk_ix_1 = chunk_distances.argmin(axis=1)
        chunk_min_distances = chunk_distances.min(axis=1)
        chunk_ix_0 = np.arange(start_idx, end_idx)
        
        all_ix_0.append(chunk_ix_0)
        all_ix_1.append(chunk_ix_1)
        all_distances.append(chunk_min_distances)
        
        # Clean up
        del chunk_distances
        gc.collect()
    
    return (
        np.concatenate(all_ix_0),
        np.concatenate(all_ix_1), 
        np.concatenate(all_distances)
    )


def stitched_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
    max_cells_for_hash: int = 50000,
    triangle_distance_threshold: float = 0.3,
    min_matching_triangles: int = 10,
    phenotype_pixel_size: float = None,
    sbs_pixel_size: float = None
) -> pd.DataFrame:
    """
    Memory-efficient alignment with adaptive triangle filtering based on actual data distribution.
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        print("Empty position data, cannot perform alignment")
        return pd.DataFrame()
    
    print(f"Starting adaptive triangle filtering alignment:")
    print(f"  Phenotype cells: {len(phenotype_positions)}")
    print(f"  SBS cells: {len(sbs_positions)}")
    print(f"  Max cells for hashing: {max_cells_for_hash}")
    
    # Analyze coordinate ranges
    phenotype_i_range = phenotype_positions['i'].max() - phenotype_positions['i'].min()
    phenotype_j_range = phenotype_positions['j'].max() - phenotype_positions['j'].min()
    sbs_i_range = sbs_positions['i'].max() - sbs_positions['i'].min()
    sbs_j_range = sbs_positions['j'].max() - sbs_positions['j'].min()
    
    empirical_scale_i = sbs_i_range / phenotype_i_range if phenotype_i_range > 0 else 1.0
    empirical_scale_j = sbs_j_range / phenotype_j_range if phenotype_j_range > 0 else 1.0
    empirical_scale = (empirical_scale_i + empirical_scale_j) / 2
    
    print(f"Coordinate analysis:")
    print(f"  Phenotype range: i={phenotype_i_range:.0f}, j={phenotype_j_range:.0f}")
    print(f"  SBS range: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")
    print(f"  Empirical scale (SBS/phenotype): {empirical_scale:.3f}")
    
    # Generate triangle hashes with subsampling
    print("\n--- Generating phenotype triangle hash ---")
    phenotype_hash = hash_cell_positions_memory_efficient(
        phenotype_positions, max_cells_for_hash
    )
    
    print("\n--- Generating SBS triangle hash ---")
    sbs_hash = hash_cell_positions_memory_efficient(
        sbs_positions, max_cells_for_hash
    )
    
    if len(phenotype_hash) == 0 or len(sbs_hash) == 0:
        print("Empty hash data, cannot perform alignment")
        return pd.DataFrame()
    
    try:
        print(f"\n--- Adaptive triangle analysis ---")
        print(f"Phenotype triangles: {len(phenotype_hash)}")
        print(f"SBS triangles: {len(sbs_hash)}")
        
        # Extract vectors and centers
        V_0, c_0 = get_vc(phenotype_hash)
        V_1, c_1 = get_vc(sbs_hash)
        
        # Analyze actual triangle magnitudes in the data
        V_0_magnitudes = np.linalg.norm(V_0, axis=1)
        V_1_magnitudes = np.linalg.norm(V_1, axis=1)
        
        print(f"Triangle magnitude analysis:")
        print(f"  Phenotype triangles:")
        print(f"    Mean: {np.mean(V_0_magnitudes):.2f}")
        print(f"    Median: {np.median(V_0_magnitudes):.2f}")
        print(f"    Std: {np.std(V_0_magnitudes):.2f}")
        print(f"    Min: {np.min(V_0_magnitudes):.2f}")
        print(f"    Max: {np.max(V_0_magnitudes):.2f}")
        print(f"    5th percentile: {np.percentile(V_0_magnitudes, 5):.2f}")
        print(f"    95th percentile: {np.percentile(V_0_magnitudes, 95):.2f}")
        
        print(f"  SBS triangles:")
        print(f"    Mean: {np.mean(V_1_magnitudes):.2f}")
        print(f"    Median: {np.median(V_1_magnitudes):.2f}")
        print(f"    Std: {np.std(V_1_magnitudes):.2f}")
        print(f"    Min: {np.min(V_1_magnitudes):.2f}")
        print(f"    Max: {np.max(V_1_magnitudes):.2f}")
        print(f"    5th percentile: {np.percentile(V_1_magnitudes, 5):.2f}")
        print(f"    95th percentile: {np.percentile(V_1_magnitudes, 95):.2f}")
        
        # Use adaptive thresholds based on actual data distribution
        # Keep top 80% of triangles from each modality
        phenotype_threshold = np.percentile(V_0_magnitudes, 20)  # Bottom 20% cutoff
        sbs_threshold = np.percentile(V_1_magnitudes, 20)  # Bottom 20% cutoff
        
        # But don't go below very small values that might be noise
        phenotype_threshold = max(phenotype_threshold, 0.1)
        sbs_threshold = max(sbs_threshold, 0.1)
        
        V_0_valid = V_0_magnitudes > phenotype_threshold
        V_1_valid = V_1_magnitudes > sbs_threshold
        
        print(f"Adaptive triangle filtering:")
        print(f"  Phenotype: {V_0_valid.sum()}/{len(V_0)} valid (threshold: {phenotype_threshold:.3f})")
        print(f"  SBS: {V_1_valid.sum()}/{len(V_1)} valid (threshold: {sbs_threshold:.3f})")
        
        if V_0_valid.sum() < 1000 or V_1_valid.sum() < 1000:
            print(f"Still insufficient triangles, trying more permissive filtering...")
            # Use top 90% instead
            phenotype_threshold = np.percentile(V_0_magnitudes, 10)
            sbs_threshold = np.percentile(V_1_magnitudes, 10)
            
            V_0_valid = V_0_magnitudes > phenotype_threshold
            V_1_valid = V_1_magnitudes > sbs_threshold
            
            print(f"  Relaxed filtering:")
            print(f"    Phenotype: {V_0_valid.sum()}/{len(V_0)} valid (threshold: {phenotype_threshold:.3f})")
            print(f"    SBS: {V_1_valid.sum()}/{len(V_1)} valid (threshold: {sbs_threshold:.3f})")
            
            if V_0_valid.sum() < 500 or V_1_valid.sum() < 500:
                print("Insufficient valid triangles even with relaxed filtering")
                return pd.DataFrame()
        
        V_0_filtered = V_0[V_0_valid]
        c_0_filtered = c_0[V_0_valid]
        V_1_filtered = V_1[V_1_valid]
        c_1_filtered = c_1[V_1_valid]
        
        # For triangle vector matching, normalize by their respective scales
        # This helps match triangles of similar relative geometry
        V_0_normalized = V_0_filtered / np.median(V_0_magnitudes[V_0_valid])
        V_1_normalized = V_1_filtered / np.median(V_1_magnitudes[V_1_valid])
        
        print(f"Triangle vector normalization:")
        print(f"  Phenotype normalization factor: {np.median(V_0_magnitudes[V_0_valid]):.3f}")
        print(f"  SBS normalization factor: {np.median(V_1_magnitudes[V_1_valid]):.3f}")
        
        # Triangle matching with chunked processing
        chunk_size = min(2000, len(V_0_normalized))  # Conservative chunk size
        print(f"Triangle matching with chunk size: {chunk_size}")
        
        i0, i1, distances = chunked_nearest_neighbors(V_0_normalized, V_1_normalized, chunk_size)
        
        # Adaptive triangle distance threshold based on normalized vectors
        # Start with provided threshold and increase if needed
        distance_threshold = triangle_distance_threshold
        
        for attempt in range(3):
            filt = distances < distance_threshold
            n_matching = filt.sum()
            
            print(f"  Attempt {attempt + 1}: threshold={distance_threshold:.3f} → {n_matching} matches")
            
            if n_matching >= min_matching_triangles:
                break
            else:
                distance_threshold *= 2  # Double the threshold and try again
        
        if n_matching < min_matching_triangles:
            print(f"Insufficient matching triangles: {n_matching} < {min_matching_triangles}")
            return pd.DataFrame()
        
        # Get matching triangle centers for transformation (use original centers)
        X, Y = c_0_filtered[i0[filt]], c_1_filtered[i1[filt]]
        
        print(f"Using {len(X)} triangle centers for transformation estimation")
        
        # Analyze the actual coordinate relationship
        print(f"\n--- Analyzing coordinate relationships ---")
        
        # Check if the coordinates are already very similar (near-identity transformation)
        coord_diff = Y - X
        diff_stats = {
            'mean': np.mean(coord_diff, axis=0),
            'std': np.std(coord_diff, axis=0),
            'median': np.median(coord_diff, axis=0),
            'max_abs': np.max(np.abs(coord_diff), axis=0)
        }
        
        print(f"Coordinate differences (Y - X):")
        print(f"  Mean: {diff_stats['mean']}")
        print(f"  Std: {diff_stats['std']}")
        print(f"  Median: {diff_stats['median']}")
        print(f"  Max absolute: {diff_stats['max_abs']}")
        
        # Check if this looks like a near-identity transformation
        mean_translation = diff_stats['mean']
        translation_magnitude = np.linalg.norm(mean_translation)
        coordinate_scale = np.mean([np.std(X), np.std(Y)])
        relative_translation = translation_magnitude / coordinate_scale if coordinate_scale > 0 else 0
        
        print(f"Translation analysis:")
        print(f"  Translation magnitude: {translation_magnitude:.2f}")
        print(f"  Coordinate scale: {coordinate_scale:.2f}")
        print(f"  Relative translation: {relative_translation:.6f}")
        
        # If the transformation appears to be primarily translation
        if relative_translation < 0.1:  # Very small relative to coordinate scale
            print("🔍 Detected near-identity transformation - using robust estimation")
            
            # Use simple median-based translation estimation
            robust_translation = np.median(coord_diff, axis=0)
            
            # Check if rotation is needed by looking at correlation
            try:
                # Subtract the translation and check if there's significant rotation
                X_centered = X - np.mean(X, axis=0)
                Y_centered = Y - np.mean(Y, axis=0) - robust_translation
                
                # Simple rotation estimation using SVD
                H = X_centered.T @ Y_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # Check if the rotation is significant
                rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                rotation_degrees = np.degrees(rotation_angle)
                
                print(f"Rotation analysis:")
                print(f"  Rotation angle: {rotation_degrees:.3f} degrees")
                print(f"  Rotation matrix determinant: {np.linalg.det(R):.6f}")
                
                if rotation_degrees < 5.0:  # Less than 5 degrees rotation
                    print("  → Using translation-only transformation")
                    final_rotation = np.eye(2)
                    final_translation = robust_translation
                    transformation_type = "translation_only"
                    determinant = 1.0
                else:
                    print("  → Using translation + rotation transformation")
                    final_rotation = R
                    final_translation = robust_translation
                    transformation_type = "translation_rotation"
                    determinant = np.linalg.det(R)
                
            except Exception as e:
                print(f"  SVD rotation estimation failed: {e}")
                print("  → Falling back to translation-only")
                final_rotation = np.eye(2)
                final_translation = robust_translation
                transformation_type = "translation_only_fallback"
                determinant = 1.0
                
        else:
            print("🔍 Non-trivial transformation detected - using RANSAC")
            
            # Continue with RANSAC approach but with adjusted parameters for near-singular cases
            
            # Check geometric diversity
            X_std = np.std(X, axis=0)
            Y_std = np.std(Y, axis=0)
            
            print(f"Triangle center spread:")
            print(f"  X: std={X_std}")
            print(f"  Y: std={Y_std}")
            
            # Use more robust RANSAC configuration for potentially ill-conditioned problems
            avg_coordinate_scale = np.mean([np.mean(X_std), np.mean(Y_std)])
            base_residual = max(50, avg_coordinate_scale * 0.02)  # More conservative
            
            # Add regularization by using a subset of well-distributed points
            if len(X) > 5000:
                # Select a geometrically diverse subset
                n_subset = 5000
                indices = np.random.choice(len(X), n_subset, replace=False)
                X_subset = X[indices]
                Y_subset = Y[indices]
                print(f"Using subset of {n_subset} points for RANSAC stability")
            else:
                X_subset = X
                Y_subset = Y
            
            # Modified RANSAC strategies for near-singular cases
            ransac_strategies = [
                # Use more samples to avoid degenerate cases
                {"min_samples": 10, "residual_threshold": base_residual, "max_trials": 3000},
                {"min_samples": 15, "residual_threshold": base_residual * 1.5, "max_trials": 4000},
                {"min_samples": 20, "residual_threshold": base_residual * 2, "max_trials": 5000},
                
                # Fall back to fewer samples with higher thresholds
                {"min_samples": 6, "residual_threshold": base_residual * 3, "max_trials": 5000},
                {"min_samples": 4, "residual_threshold": base_residual * 5, "max_trials": 8000},
                {"min_samples": 3, "residual_threshold": base_residual * 10, "max_trials": 10000},
            ]
            
            print(f"RANSAC base residual threshold: {base_residual:.1f}")
            
            model = None
            transformation_type = "full"
            
            for i, config in enumerate(ransac_strategies):
                try:
                    print(f"RANSAC strategy {i+1}: min_samples={config['min_samples']}, residual_threshold={config['residual_threshold']:.1f}")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        test_model = RANSACRegressor(**config)
                        test_model.fit(X_subset, Y_subset)
                    
                    rotation = test_model.estimator_.coef_
                    determinant = np.linalg.det(rotation)
                    
                    print(f"  Result: determinant={determinant:.6f}")
                    
                    # More lenient determinant check
                    if abs(determinant) > 0.1:
                        model = test_model
                        transformation_type = "full"
                        print(f"  ✅ Accepted (determinant > 0.1)")
                        break
                    elif abs(determinant) > 0.01:
                        model = test_model
                        transformation_type = "low_confidence"
                        print(f"  ⚠️  Accepted with low confidence (determinant > 0.01)")
                        break
                    else:
                        print(f"  ❌ Determinant too close to zero")
                        
                except Exception as e:
                    print(f"  ❌ RANSAC failed: {e}")
                    continue
            
            if model is None:
                print("All RANSAC strategies failed, falling back to robust translation estimation")
                # Final fallback to translation-only
                final_rotation = np.eye(2)
                final_translation = np.median(coord_diff, axis=0)
                transformation_type = "translation_fallback"
                determinant = 1.0
            else:
                final_rotation = model.estimator_.coef_
                final_translation = model.estimator_.intercept_
                determinant = np.linalg.det(final_rotation)
        
        # Score the final transformation
        print(f"\n--- Scoring final transformation ---")
        print(f"Transformation type: {transformation_type}")
        print(f"Rotation determinant: {determinant:.6f}")
        print(f"Translation: {final_translation}")
        
        # Apply transformation and score
        n_score = min(2000, len(X))
        if n_score < len(X):
            score_indices = np.random.choice(len(X), n_score, replace=False)
            X_score = X[score_indices]
        else:
            X_score = X
        
        # Apply the transformation
        if transformation_type.startswith("translation"):
            predicted_centers = X_score + final_translation
        else:
            predicted_centers = X_score @ final_rotation.T + final_translation
        
        tree = cKDTree(c_1_filtered)
        distances_score, _ = tree.query(predicted_centers)
        
        # Adaptive scoring thresholds based on transformation type
        if transformation_type.startswith("translation"):
            # More lenient for translation-only
            score_threshold_region = coordinate_scale * 0.5
            score_threshold_match = coordinate_scale * 0.25
        else:
            # Standard thresholds for full transformation
            score_threshold_region = coordinate_scale * 1.0
            score_threshold_match = coordinate_scale * 0.5
        
        filt_score = distances_score < score_threshold_region
        score = (distances_score[filt_score] < score_threshold_match).mean() if filt_score.sum() > 0 else 0
        
        print(f"Scoring thresholds: region={score_threshold_region:.1f}, match={score_threshold_match:.1f}")
        print(f"Score: {score:.3f} ({filt_score.sum()}/{len(distances_score)} centers in region)")
        
        # Create result
        result = pd.DataFrame([{
            'rotation_1': final_rotation[0] if final_rotation.shape[0] > 0 else [1, 0],
            'rotation_2': final_rotation[1] if final_rotation.shape[0] > 1 else [0, 1], 
            'translation': final_translation,
            'score': score,
            'determinant': determinant,
            'well': phenotype_positions['well'].iloc[0] if len(phenotype_positions) > 0 else 'unknown',
            'n_triangles_matched': n_matching,
            'n_triangles_phenotype': len(c_0_filtered),
            'n_triangles_sbs': len(c_1_filtered),
            'cells_used_phenotype': len(phenotype_hash),
            'cells_used_sbs': len(sbs_hash),
            'empirical_scale_factor': empirical_scale,
            'transformation_type': transformation_type,
            'relative_translation': relative_translation,
            'coordinate_scale': coordinate_scale,
            'phenotype_pixel_size': phenotype_pixel_size,
            'sbs_pixel_size': sbs_pixel_size
        }])
        
        print(f"✅ Near-identity transformation alignment results:")
        print(f"   Score: {score:.3f}")
        print(f"   Determinant: {determinant:.6f}")
        print(f"   Transformation type: {transformation_type}")
        print(f"   Translation: {final_translation}")
        print(f"   Relative translation: {relative_translation:.6f}")
        print(f"   Matched triangles: {n_matching}")
        print(f"   Cells used: {len(phenotype_hash)} phenotype, {len(sbs_hash)} SBS")
        
        return result
        
        # If we get here, try RANSAC with the (possibly improved) triangle selection
        avg_coordinate_scale = (np.mean(X_std) + np.mean(Y_std)) / 2
        base_residual = max(100, avg_coordinate_scale * 0.05)  # More conservative scaling
        
        # Try different RANSAC strategies
        ransac_strategies = [
            # Strategy 1: Standard RANSAC with conservative thresholds
            {"min_samples": 3, "residual_threshold": base_residual, "max_trials": 2000},
            {"min_samples": 4, "residual_threshold": base_residual * 1.5, "max_trials": 3000},
            {"min_samples": 5, "residual_threshold": base_residual * 2, "max_trials": 4000},
            
            # Strategy 2: More permissive residuals
            {"min_samples": 3, "residual_threshold": base_residual * 3, "max_trials": 3000},
            {"min_samples": 3, "residual_threshold": base_residual * 5, "max_trials": 5000},
            
            # Strategy 3: Very permissive as last resort
            {"min_samples": 3, "residual_threshold": base_residual * 10, "max_trials": 8000},
        ]
        
        print(f"RANSAC base residual threshold: {base_residual:.1f}")
        
        model = None
        transformation_type = "full"
        
        for i, config in enumerate(ransac_strategies):
            try:
                print(f"RANSAC strategy {i+1}: min_samples={config['min_samples']}, residual_threshold={config['residual_threshold']:.1f}")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    test_model = RANSACRegressor(**config)
                    test_model.fit(X, Y)
                
                rotation = test_model.estimator_.coef_
                determinant = np.linalg.det(rotation)
                
                print(f"  Result: determinant={determinant:.6f}")
                
                # More permissive determinant check - accept if not too close to zero
                if abs(determinant) > 0.01:  # Increased from 0.001
                    model = test_model
                    print(f"  ✅ Accepted (determinant > 0.01)")
                    break
                elif abs(determinant) > 0.001:  # Try to use it anyway with warning
                    model = test_model
                    transformation_type = "low_confidence"
                    print(f"  ⚠️  Accepted with low confidence (determinant > 0.001)")
                    break
                else:
                    print(f"  ❌ Determinant too close to zero")
                    
            except Exception as e:
                print(f"  ❌ RANSAC failed: {e}")
                continue
        
        if model is None:
            print("All RANSAC strategies failed")
            return pd.DataFrame()
        
        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_
        determinant = np.linalg.det(rotation)
        
        # Score the transformation
        print("--- Scoring transformation ---")
        
        n_score = min(1000, len(X))
        if n_score < len(X):
            score_indices = np.random.choice(len(X), n_score, replace=False)
            X_score = X[score_indices]
        else:
            X_score = X
        
        predicted_centers = X_score @ rotation.T + translation
        tree = cKDTree(c_1_filtered)
        distances_score, _ = tree.query(predicted_centers)
        
        # Scale-aware scoring thresholds
        score_threshold_region = avg_coordinate_scale * 1.5
        score_threshold_match = avg_coordinate_scale * 0.75
        
        filt_score = distances_score < score_threshold_region
        score = (distances_score[filt_score] < score_threshold_match).mean() if filt_score.sum() > 0 else 0
        
        print(f"Scoring thresholds: region={score_threshold_region:.1f}, match={score_threshold_match:.1f}")
        print(f"Score calculation: {filt_score.sum()}/{len(distances_score)} centers in region, score={score:.3f}")
        
        # Create result
        result = pd.DataFrame([{
            'rotation_1': rotation[0] if len(rotation) > 0 else [0, 0],
            'rotation_2': rotation[1] if len(rotation) > 1 else [0, 0], 
            'translation': translation,
            'score': score,
            'determinant': determinant,
            'well': phenotype_positions['well'].iloc[0] if len(phenotype_positions) > 0 else 'unknown',
            'n_triangles_matched': n_matching,
            'n_triangles_phenotype': len(c_0_filtered),
            'n_triangles_sbs': len(c_1_filtered),
            'cells_used_phenotype': len(phenotype_hash),
            'cells_used_sbs': len(sbs_hash),
            'empirical_scale_factor': empirical_scale,
            'phenotype_triangle_threshold': phenotype_threshold,
            'sbs_triangle_threshold': sbs_threshold,
            'distance_threshold_used': distance_threshold,
            'transformation_type': transformation_type,
            'x_aspect_ratio': x_aspect_ratio,
            'y_aspect_ratio': y_aspect_ratio,
            'phenotype_pixel_size': phenotype_pixel_size,
            'sbs_pixel_size': sbs_pixel_size
        }])
        
        print(f"✅ Geometric diversity-aware alignment results:")
        print(f"   Score: {score:.3f}")
        print(f"   Determinant: {determinant:.6f}")
        print(f"   Transformation type: {transformation_type}")
        print(f"   Empirical scale factor: {empirical_scale:.3f}")
        print(f"   Aspect ratios: X={x_aspect_ratio:.2f}, Y={y_aspect_ratio:.2f}")
        print(f"   Matched triangles: {n_matching}")
        print(f"   Cells used: {len(phenotype_hash)} phenotype, {len(sbs_hash)} SBS")
        
        return result
        
    except Exception as e:
        print(f"Adaptive triangle filtering alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def subsample_cells_for_alignment(
    cell_positions: pd.DataFrame, 
    max_cells: int = 50000,
    spatial_bins: int = 15,  # Increased from 10 to 15 for better distribution
    random_state: int = 42
) -> pd.DataFrame:
    """
    Subsample cells while preserving spatial distribution.
    
    Args:
        cell_positions: DataFrame with cell positions
        max_cells: Maximum number of cells to keep
        spatial_bins: Number of spatial bins per dimension for stratified sampling
        random_state: Random seed for reproducibility
        
    Returns:
        Subsampled DataFrame
    """
    if len(cell_positions) <= max_cells:
        return cell_positions
    
    print(f"Subsampling from {len(cell_positions)} to {max_cells} cells")
    
    # Create spatial bins with some padding to avoid edge effects
    i_min, i_max = cell_positions['i'].min(), cell_positions['i'].max()
    j_min, j_max = cell_positions['j'].min(), cell_positions['j'].max()
    
    # Add 5% padding to avoid edge effects
    i_range = i_max - i_min
    j_range = j_max - j_min
    i_bins = pd.cut(
        cell_positions['i'], 
        bins=np.linspace(i_min - 0.05*i_range, i_max + 0.05*i_range, spatial_bins + 1), 
        labels=False,
        include_lowest=True
    )
    j_bins = pd.cut(
        cell_positions['j'], 
        bins=np.linspace(j_min - 0.05*j_range, j_max + 0.05*j_range, spatial_bins + 1), 
        labels=False,
        include_lowest=True
    )
    
    cell_positions = cell_positions.copy()
    cell_positions['spatial_bin'] = i_bins * spatial_bins + j_bins
    
    # Remove any NaN bins
    cell_positions = cell_positions.dropna(subset=['spatial_bin'])
    
    # Stratified sampling within each spatial bin
    np.random.seed(random_state)
    
    subsampled_parts = []
    cells_per_bin = max_cells // (spatial_bins * spatial_bins)
    
    # Also add some random sampling to fill up to max_cells
    total_sampled = 0
    
    for bin_id in range(spatial_bins * spatial_bins):
        bin_cells = cell_positions[cell_positions['spatial_bin'] == bin_id]
        if len(bin_cells) > 0:
            n_sample = min(len(bin_cells), max(cells_per_bin, 1))
            subsampled = bin_cells.sample(n=n_sample, random_state=random_state + bin_id)
            subsampled_parts.append(subsampled)
            total_sampled += len(subsampled)
    
    if subsampled_parts:
        result = pd.concat(subsampled_parts, ignore_index=True)
        result = result.drop('spatial_bin', axis=1)
        
        # If we're still under max_cells, add some random additional cells
        if total_sampled < max_cells and total_sampled < len(cell_positions):
            remaining_cells = max_cells - total_sampled
            # Get cells not already selected
            selected_indices = set(result.index) if hasattr(result, 'index') else set()
            available_cells = cell_positions[~cell_positions.index.isin(selected_indices)]
            
            if len(available_cells) > 0:
                additional_sample_size = min(remaining_cells, len(available_cells))
                additional_cells = available_cells.sample(
                    n=additional_sample_size, 
                    random_state=random_state + 999
                ).drop('spatial_bin', axis=1)
                result = pd.concat([result, additional_cells], ignore_index=True)
        
        print(f"Subsampled to {len(result)} cells")
        return result
    else:
        print("No cells found in any spatial bin, using random sampling")
        # Fallback to simple random sampling
        sample_size = min(max_cells, len(cell_positions))
        return cell_positions.sample(n=sample_size, random_state=random_state)


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame, 
    alignment: pd.Series,
    threshold: float = 2.0,
    chunk_size: int = 50000
) -> pd.DataFrame:
    """
    Memory-efficient cell merging using chunked processing.
    
    Args:
        phenotype_positions: Cell positions in phenotype well
        sbs_positions: Cell positions in SBS well
        alignment: Alignment parameters
        threshold: Maximum distance for cell matching
        chunk_size: Size of chunks for processing
        
    Returns:
        DataFrame with merged cell identities
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        return pd.DataFrame(columns=[
            'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
    
    try:
        print(f"Memory-efficient cell merging:")
        print(f"  Phenotype cells: {len(phenotype_positions)}")
        print(f"  SBS cells: {len(sbs_positions)}")
        print(f"  Chunk size: {chunk_size}")
        
        # Build transformation model
        rotation = np.array([alignment['rotation_1'], alignment['rotation_2']])
        translation = alignment['translation']
        model = LinearRegression()
        model.coef_ = rotation
        model.intercept_ = translation
        
        # Extract coordinates
        X = phenotype_positions[['i', 'j']].values
        Y = sbs_positions[['i', 'j']].values
        
        print("Applying transformation...")
        X_transformed = model.predict(X)
        
        print("Building SBS KDTree for efficient nearest neighbor search...")
        tree = cKDTree(Y)
        
        # Process in chunks to avoid memory issues
        all_matches = []
        n_chunks = (len(X_transformed) + chunk_size - 1) // chunk_size
        
        print(f"Processing {len(X_transformed)} cells in {n_chunks} chunks...")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(X_transformed))
            
            print(f"  Chunk {chunk_idx + 1}/{n_chunks}: cells {start_idx}-{end_idx}")
            
            # Get chunk of transformed coordinates
            X_chunk = X_transformed[start_idx:end_idx]
            
            # Find nearest neighbors using KDTree
            distances, indices = tree.query(X_chunk)
            
            # Filter by threshold
            within_threshold = distances < threshold
            
            if within_threshold.sum() == 0:
                continue
            
            # Get matching data
            chunk_phenotype = phenotype_positions.iloc[start_idx:end_idx][within_threshold].reset_index(drop=True)
            chunk_sbs = sbs_positions.iloc[indices[within_threshold]].reset_index(drop=True)
            chunk_distances = distances[within_threshold]
            
            # Create chunk results
            chunk_matches = pd.DataFrame({
                'plate': 1,  # You may need to extract this from your data
                'well': chunk_phenotype['well'] if 'well' in chunk_phenotype.columns else alignment.get('well', 'unknown'),
                'cell_0': chunk_phenotype['cell'],
                'i_0': chunk_phenotype['i'],
                'j_0': chunk_phenotype['j'], 
                'area_0': chunk_phenotype['area'],
                'cell_1': chunk_sbs['cell'],
                'i_1': chunk_sbs['i'],
                'j_1': chunk_sbs['j'],
                'area_1': chunk_sbs['area'],
                'distance': chunk_distances
            })
            
            all_matches.append(chunk_matches)
            
            # Clean up
            del X_chunk, distances, indices
            gc.collect()
        
        if not all_matches:
            print("No matches found within threshold")
            return pd.DataFrame(columns=[
                'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
                'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
            ])
        
        # Combine all chunks
        merged_data = pd.concat(all_matches, ignore_index=True)
        
        print(f"✅ Successfully merged {len(merged_data)} cells")
        print(f"   Mean distance: {merged_data['distance'].mean():.2f}")
        print(f"   Max distance: {merged_data['distance'].max():.2f}")
        
        return merged_data
        
    except Exception as e:
        print(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=[
            'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])