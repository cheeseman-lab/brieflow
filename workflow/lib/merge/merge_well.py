"""
Enhanced well-level merge functions for stitched image data.

This module handles alignment and merging of phenotype and SBS data at the well level
after images have been stitched, accounting for magnification differences and coordinate
transformations.
"""

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
from typing import Optional, Tuple, Dict, Any


def calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size):
    """
    Calculate expected scale factor with correct physics.
    
    For your setup:
    - Phenotype: 40x magnification, no binning, 0.1625 μm/pixel
    - SBS: 10x magnification, 2x2 binning, 1.3 μm/pixel
    
    Expected scale = SBS_coordinate_range / Phenotype_coordinate_range
    Should be ~1/8 = 0.125 because SBS covers same physical area with fewer pixels
    """
    if phenotype_pixel_size and sbs_pixel_size:
        # This gives us the ratio of physical distances per pixel
        # SBS pixels represent more physical distance, so coordinates are smaller
        expected_scale = phenotype_pixel_size / sbs_pixel_size
        return expected_scale
    return None


def subsample_cells_for_hashing(positions_df: pd.DataFrame, max_cells: int = 75000) -> pd.DataFrame:
    """
    Subsample cells for triangle hashing if dataset is too large.
    
    Args:
        positions_df: DataFrame with cell positions
        max_cells: Maximum number of cells to use for hashing
        
    Returns:
        Subsampled DataFrame
    """
    if len(positions_df) <= max_cells:
        return positions_df
    
    print(f"Subsampling {len(positions_df):,} cells to {max_cells:,} for triangle hashing")
    
    # Use systematic sampling to maintain spatial distribution
    step = len(positions_df) // max_cells
    indices = np.arange(0, len(positions_df), step)[:max_cells]
    
    return positions_df.iloc[indices].copy()


def generate_triangle_hash(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate triangle hash from cell positions using Delaunay triangulation.
    
    Args:
        positions_df: DataFrame with 'i', 'j' columns for coordinates
        
    Returns:
        DataFrame with triangle features for matching
    """
    if len(positions_df) < 4:
        return pd.DataFrame()
    
    # Extract coordinates
    coords = positions_df[['i', 'j']].values
    
    try:
        # Create Delaunay triangulation
        tri = Delaunay(coords)
        
        triangles = []
        for simplex_idx, simplex in enumerate(tri.simplices):
            # Skip triangles on the boundary (have neighbors = -1)
            if (tri.neighbors[simplex_idx] == -1).any():
                continue
                
            # Get triangle vertices
            vertices = coords[simplex]
            
            # Calculate triangle features
            # Edge lengths
            edge1 = np.linalg.norm(vertices[1] - vertices[0])
            edge2 = np.linalg.norm(vertices[2] - vertices[1]) 
            edge3 = np.linalg.norm(vertices[0] - vertices[2])
            
            # Sort edges for rotation invariance
            edges = sorted([edge1, edge2, edge3])
            
            # Calculate triangle center
            center = vertices.mean(axis=0)
            
            # Calculate area
            area = 0.5 * abs(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
            
            # Create normalized feature vector
            if edges[2] > 0:  # Avoid division by zero
                features = {
                    'edge_ratio_1': edges[0] / edges[2],
                    'edge_ratio_2': edges[1] / edges[2], 
                    'area_normalized': area / (edges[2] ** 2),
                    'center_i': center[0],
                    'center_j': center[1],
                    'max_edge': edges[2]
                }
                triangles.append(features)
        
        return pd.DataFrame(triangles)
    
    except Exception as e:
        print(f"Triangle hash generation failed: {e}")
        return pd.DataFrame()


def find_triangle_matches(hash1: pd.DataFrame, hash2: pd.DataFrame, 
                         distance_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find matching triangles between two triangle hash DataFrames.
    
    Args:
        hash1, hash2: Triangle hash DataFrames
        distance_threshold: Maximum distance for triangle matching
        
    Returns:
        Tuple of (indices1, indices2, distances) for matching triangles
    """
    if len(hash1) == 0 or len(hash2) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Extract feature vectors (edge ratios and normalized area)
    features1 = hash1[['edge_ratio_1', 'edge_ratio_2', 'area_normalized']].values
    features2 = hash2[['edge_ratio_1', 'edge_ratio_2', 'area_normalized']].values
    
    # Find nearest neighbors in feature space
    distances = cdist(features1, features2, metric='euclidean')
    
    # Find best matches
    idx1 = np.arange(len(features1))
    idx2 = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    # Filter by distance threshold
    valid = min_distances < distance_threshold
    
    return idx1[valid], idx2[valid], min_distances[valid]


def robust_transformation_estimation(centers1: np.ndarray, centers2: np.ndarray,
                                   transformation_type: str = 'auto') -> Dict[str, Any]:
    """
    Estimate transformation between two sets of triangle centers using RANSAC.
    
    Args:
        centers1, centers2: Corresponding triangle centers
        transformation_type: 'translation_only', 'translation_rotation', or 'auto'
        
    Returns:
        Dictionary with transformation parameters and quality metrics
    """
    if len(centers1) < 3:
        return {
            'rotation': np.eye(2),
            'translation': np.zeros(2),
            'score': 0.0,
            'determinant': 1.0,
            'transformation_type': 'failed',
            'n_triangles_matched': len(centers1)
        }
    
    best_result = None
    best_score = -1
    
    # Try different transformation types
    if transformation_type == 'auto':
        transformation_types = ['translation_only', 'translation_rotation']
    else:
        transformation_types = [transformation_type]
    
    for trans_type in transformation_types:
        try:
            if trans_type == 'translation_only':
                # Simple translation-only transformation
                translation = np.median(centers2 - centers1, axis=0)
                rotation = np.eye(2)
                
                # Calculate score
                predicted = centers1 + translation
                distances = np.sqrt(((predicted - centers2) ** 2).sum(axis=1))
                score = (distances < 5.0).mean()  # Within 5 pixels
                
                result = {
                    'rotation': rotation,
                    'translation': translation,
                    'score': score,
                    'determinant': 1.0,
                    'transformation_type': trans_type,
                    'n_triangles_matched': len(centers1)
                }
                
            else:  # translation_rotation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    
                    # Use RANSAC for robust estimation
                    if len(centers1) >= 10:
                        ransac = RANSACRegressor(
                            min_samples=max(3, len(centers1) // 4),
                            residual_threshold=3.0,
                            max_trials=1000
                        )
                    else:
                        ransac = RANSACRegressor(min_samples=len(centers1))
                    
                    ransac.fit(centers1, centers2)
                    
                    rotation = ransac.estimator_.coef_
                    translation = ransac.estimator_.intercept_
                    
                    # Calculate score based on inliers
                    predicted = centers1 @ rotation.T + translation
                    distances = np.sqrt(((predicted - centers2) ** 2).sum(axis=1))
                    score = (distances < 3.0).mean()
                    
                    determinant = np.linalg.det(rotation)
                    
                    result = {
                        'rotation': rotation,
                        'translation': translation,
                        'score': score,
                        'determinant': determinant,
                        'transformation_type': trans_type,
                        'n_triangles_matched': len(centers1)
                    }
            
            # Keep best result
            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                
        except Exception as e:
            print(f"Transformation estimation failed for {trans_type}: {e}")
            continue
    
    if best_result is None:
        return {
            'rotation': np.eye(2),
            'translation': np.zeros(2),
            'score': 0.0,
            'determinant': 1.0,
            'transformation_type': 'failed',
            'n_triangles_matched': 0
        }
    
    return best_result


def stitched_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.5, 2.0),
    score_threshold: float = 0.1,
    max_cells_for_hash: int = 75000,
    triangle_distance_threshold: float = 0.3,
    min_matching_triangles: int = 10,
    phenotype_pixel_size: Optional[float] = None,
    sbs_pixel_size: Optional[float] = None
) -> pd.DataFrame:
    """
    Perform triangle hash alignment between stitched phenotype and SBS data.
    
    Args:
        phenotype_positions: DataFrame with phenotype cell positions (i, j columns)
        sbs_positions: DataFrame with SBS cell positions (i, j columns)
        det_range: Acceptable range for transformation determinant
        score_threshold: Minimum score for valid alignment
        max_cells_for_hash: Maximum cells to use for triangle hashing
        triangle_distance_threshold: Maximum distance for triangle matching
        min_matching_triangles: Minimum triangles needed for alignment
        phenotype_pixel_size: Pixel size for phenotype data (μm/pixel)
        sbs_pixel_size: Pixel size for SBS data (μm/pixel)
        
    Returns:
        DataFrame with alignment parameters
    """
    print(f"Starting stitched well alignment with {len(phenotype_positions):,} phenotype and {len(sbs_positions):,} SBS cells")
    
    # Check minimum requirements
    if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
        print("Insufficient cells for triangulation")
        return pd.DataFrame()
    
    # Subsample if necessary
    pheno_subset = subsample_cells_for_hashing(phenotype_positions, max_cells_for_hash)
    sbs_subset = subsample_cells_for_hashing(sbs_positions, max_cells_for_hash)
    
    print(f"Using {len(pheno_subset):,} phenotype and {len(sbs_subset):,} SBS cells for alignment")
    
    # Generate triangle hashes
    print("Generating triangle hashes...")
    pheno_triangles = generate_triangle_hash(pheno_subset)
    sbs_triangles = generate_triangle_hash(sbs_subset)
    
    if len(pheno_triangles) == 0 or len(sbs_triangles) == 0:
        print("Failed to generate triangle hashes")
        return pd.DataFrame()
    
    print(f"Generated {len(pheno_triangles)} phenotype and {len(sbs_triangles)} SBS triangles")
    
    # Find matching triangles
    print("Finding triangle matches...")
    idx1, idx2, distances = find_triangle_matches(
        pheno_triangles, sbs_triangles, triangle_distance_threshold
    )
    
    if len(idx1) < min_matching_triangles:
        print(f"Insufficient triangle matches: {len(idx1)} < {min_matching_triangles}")
        return pd.DataFrame()
    
    print(f"Found {len(idx1)} matching triangles")
    
    # Extract matching triangle centers
    pheno_centers = pheno_triangles.iloc[idx1][['center_i', 'center_j']].values
    sbs_centers = sbs_triangles.iloc[idx2][['center_i', 'center_j']].values
    
    # Estimate transformation
    print("Estimating transformation...")
    alignment = robust_transformation_estimation(pheno_centers, sbs_centers)
    
    # Add metadata
    alignment['cells_used_phenotype'] = len(pheno_subset)
    alignment['cells_used_sbs'] = len(sbs_subset)
    alignment['triangles_generated_phenotype'] = len(pheno_triangles)
    alignment['triangles_generated_sbs'] = len(sbs_triangles)
    alignment['triangles_matched'] = len(idx1)
    
    # Add pixel size information
    if phenotype_pixel_size and sbs_pixel_size:
        expected_scale = calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size)
        alignment['expected_scale_factor'] = expected_scale
        
        # Check if coordinates appear scale-normalized
        pheno_range = (phenotype_positions['i'].max() - phenotype_positions['i'].min() + 
                      phenotype_positions['j'].max() - phenotype_positions['j'].min()) / 2
        sbs_range = (sbs_positions['i'].max() - sbs_positions['i'].min() + 
                    sbs_positions['j'].max() - sbs_positions['j'].min()) / 2
        
        if sbs_range > 0:
            empirical_scale = pheno_range / sbs_range
            alignment['empirical_scale_factor'] = empirical_scale
            alignment['scale_normalized'] = abs(empirical_scale - 1.0) < 0.3
        else:
            alignment['empirical_scale_factor'] = None
            alignment['scale_normalized'] = False
    else:
        alignment['expected_scale_factor'] = None
        alignment['empirical_scale_factor'] = None
        alignment['scale_normalized'] = None
    
    # Convert to DataFrame
    result_df = pd.DataFrame([alignment])
    
    print(f"Alignment result: score={alignment['score']:.3f}, determinant={alignment['determinant']:.3f}, type={alignment['transformation_type']}")
    
    return result


# Legacy compatibility functions - keep these for backwards compatibility
def build_linear_model(rotation, translation):
    """Builds a linear regression model using the provided rotation matrix and translation vector."""
    from sklearn.linear_model import LinearRegression
    
    m = LinearRegression()
    m.coef_ = rotation
    m.intercept_ = translation
    return m


def merge_sbs_phenotype(cell_locations_0, cell_locations_1, model, threshold=2):
    """Legacy function for backwards compatibility with tile-by-tile approach."""
    # This is kept for compatibility but the enhanced approach uses merge_stitched_cells
    
    # Final columns for the merged DataFrame
    cols_final = [
        "plate", "well", "tile", "cell_0", "i_0", "j_0", 
        "site", "cell_1", "i_1", "j_1", "distance"
    ]

    # Check if either dataframe is None or empty
    if (cell_locations_0 is None or cell_locations_1 is None or 
        (hasattr(cell_locations_0, "empty") and cell_locations_0.empty) or
        (hasattr(cell_locations_1, "empty") and cell_locations_1.empty)):
        return pd.DataFrame(columns=cols_final)

    # Extract coordinates from the DataFrames
    X = cell_locations_0[["i", "j"]].values
    Y = cell_locations_1[["i", "j"]].values

    # Predict coordinates for dataset 0 using the alignment model
    Y_pred = model.predict(X)

    # Calculate squared Euclidean distances
    distances = cdist(Y, Y_pred, metric="sqeuclidean")

    # Find the index of the nearest neighbor for each point
    ix = distances.argmin(axis=1)

    # Filter matches based on the threshold distance
    filt = np.sqrt(distances.min(axis=1)) < threshold

    # Define new column names for merging
    columns_0 = {"tile": "tile", "cell": "cell_0", "i": "i_0", "j": "j_0"}
    columns_1 = {"site": "site", "cell": "cell_1", "i": "i_1", "j": "j_1"}

    # Prepare the target DataFrame with matched coordinates
    target = (
        cell_locations_0.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)
    )

    # Merge DataFrames and calculate distances
    return (
        cell_locations_1[filt]
        .reset_index(drop=True)[list(columns_1.keys())]
        .rename(columns=columns_1)
        .pipe(lambda x: pd.concat([target, x], axis=1))
    .assign(distance=np.sqrt(distances.min(axis=1))[filt])[cols_final]
    )


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 2.0,
    chunk_size: int = 50000
) -> pd.DataFrame:
    """
    Merge cells using alignment transformation with memory-efficient processing.
    
    Args:
        phenotype_positions: DataFrame with phenotype cell positions
        sbs_positions: DataFrame with SBS cell positions  
        alignment: Alignment parameters from stitched_well_alignment
        threshold: Maximum distance for cell matching (pixels)
        chunk_size: Process cells in chunks to manage memory
        
    Returns:
        DataFrame with merged cell information
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


def corrected_scale_analysis(phenotype_positions, sbs_positions, 
                           phenotype_pixel_size, sbs_pixel_size):
    """
    Corrected scale analysis with proper physics understanding.
    """
    # Calculate coordinate ranges
    pheno_i_range = phenotype_positions['i'].max() - phenotype_positions['i'].min()
    pheno_j_range = phenotype_positions['j'].max() - phenotype_positions['j'].min()
    sbs_i_range = sbs_positions['i'].max() - sbs_positions['i'].min()
    sbs_j_range = sbs_positions['j'].max() - sbs_positions['j'].min()
    
    # Empirical scale (what we observe)
    empirical_scale_i = sbs_i_range / pheno_i_range if pheno_i_range > 0 else 1.0
    empirical_scale_j = sbs_j_range / pheno_j_range if pheno_j_range > 0 else 1.0
    empirical_scale = (empirical_scale_i + empirical_scale_j) / 2
    
    # Expected scale (what physics tells us)
    expected_scale = calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size)
    
    print(f"=== CORRECTED Scale Analysis ===")
    print(f"Coordinate ranges:")
    print(f"  Phenotype: i={pheno_i_range:.0f}, j={pheno_j_range:.0f}")
    print(f"  SBS: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")
    print(f"Empirical scale factor (SBS/phenotype): {empirical_scale:.3f}")
    
    if expected_scale:
        print(f"Expected scale factor: {expected_scale:.3f}")
        scale_difference = abs(empirical_scale - expected_scale)
        print(f"Scale difference: {scale_difference:.3f}")
        
        # Check if scales match (within 50% tolerance)
        if scale_difference < expected_scale * 0.5:
            print("✅ Scales match expected physics - coordinates are correct!")
            return {
                'scales_match': True,
                'empirical_scale': empirical_scale,
                'expected_scale': expected_scale,
                'scale_difference': scale_difference,
                'coordinate_correction_needed': False
            }
        else:
            print("⚠️ Scale mismatch detected")
            return {
                'scales_match': False,
                'empirical_scale': empirical_scale,
                'expected_scale': expected_scale,
                'scale_difference': scale_difference,
                'coordinate_correction_needed': True
            }
    else:
        print("No pixel size metadata available")
        return {
            'scales_match': None,
            'empirical_scale': empirical_scale,
            'expected_scale': None,
            'coordinate_correction_needed': False
        }


def analyze_overlap_with_correct_scale(phenotype_positions, sbs_positions, alignment):
    """
    Analyze coordinate overlap with correct scale understanding.
    """
    print(f"\n=== Overlap Analysis with Correct Scale ===")
    
    # Apply the transformation to phenotype coordinates
    rotation = np.array(alignment['rotation']).reshape(2, 2)
    translation = alignment['translation']
    
    # Transform phenotype coordinates to SBS coordinate system
    pheno_coords = phenotype_positions[['i', 'j']].values
    transformed_coords = pheno_coords @ rotation.T + translation
    
    # Find overlap region in SBS coordinate system
    sbs_coords = sbs_positions[['i', 'j']].values
    
    # Calculate bounds
    sbs_i_min, sbs_i_max = sbs_coords[:, 0].min(), sbs_coords[:, 0].max()
    sbs_j_min, sbs_j_max = sbs_coords[:, 1].min(), sbs_coords[:, 1].max()
    
    transformed_i_min, transformed_i_max = transformed_coords[:, 0].min(), transformed_coords[:, 0].max()
    transformed_j_min, transformed_j_max = transformed_coords[:, 1].min(), transformed_coords[:, 1].max()
    
    # Find intersection
    overlap_i_min = max(sbs_i_min, transformed_i_min)
    overlap_i_max = min(sbs_i_max, transformed_i_max)
    overlap_j_min = max(sbs_j_min, transformed_j_min)
    overlap_j_max = min(sbs_j_max, transformed_j_max)
    
    print(f"Coordinate bounds (in SBS coordinate system):")
    print(f"  SBS: i=[{sbs_i_min:.0f}, {sbs_i_max:.0f}], j=[{sbs_j_min:.0f}, {sbs_j_max:.0f}]")
    print(f"  Transformed phenotype: i=[{transformed_i_min:.0f}, {transformed_i_max:.0f}], j=[{transformed_j_min:.0f}, {transformed_j_max:.0f}]")
    print(f"  Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
    
    # Check if there's meaningful overlap
    if overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_total_area = (sbs_i_max - sbs_i_min) * (sbs_j_max - sbs_j_min)
        overlap_fraction = overlap_area / sbs_total_area if sbs_total_area > 0 else 0
        
        print(f"  Overlap area: {overlap_area:.0f} square pixels ({overlap_fraction:.1%} of SBS area)")
        
        # Count cells in overlap region
        sbs_in_overlap = np.sum(
            (sbs_coords[:, 0] >= overlap_i_min) & (sbs_coords[:, 0] <= overlap_i_max) &
            (sbs_coords[:, 1] >= overlap_j_min) & (sbs_coords[:, 1] <= overlap_j_max)
        )
        
        transformed_in_overlap = np.sum(
            (transformed_coords[:, 0] >= overlap_i_min) & (transformed_coords[:, 0] <= overlap_i_max) &
            (transformed_coords[:, 1] >= overlap_j_min) & (transformed_coords[:, 1] <= overlap_j_max)
        )
        
        print(f"Cells in overlap region:")
        print(f"  SBS: {sbs_in_overlap:,} ({100*sbs_in_overlap/len(sbs_positions):.1f}%)")
        print(f"  Phenotype: {transformed_in_overlap:,} ({100*transformed_in_overlap/len(phenotype_positions):.1f}%)")
        print(f"  Expected max matches: {min(sbs_in_overlap, transformed_in_overlap):,}")
        
        return {
            'has_overlap': True,
            'overlap_fraction': overlap_fraction,
            'sbs_cells_in_overlap': sbs_in_overlap,
            'phenotype_cells_in_overlap': transformed_in_overlap,
            'expected_max_matches': min(sbs_in_overlap, transformed_in_overlap)
        }
    else:
        print("❌ No meaningful overlap detected!")
        return {
            'has_overlap': False,
            'overlap_fraction': 0,
            'sbs_cells_in_overlap': 0,
            'phenotype_cells_in_overlap': 0,
            'expected_max_matches': 0
        }


def enhanced_alignment_with_correct_scale(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    phenotype_pixel_size: float = None,
    sbs_pixel_size: float = None,
    **kwargs
) -> pd.DataFrame:
    """
    Enhanced alignment that understands the correct scale relationship.
    """
    print("=== Enhanced Alignment with Correct Scale Physics ===")
    
    # Step 1: Verify scale is as expected
    scale_analysis = corrected_scale_analysis(
        phenotype_positions, sbs_positions,
        phenotype_pixel_size, sbs_pixel_size
    )
    
    # Step 2: Run alignment (coordinates should already be correct)
    result = stitched_well_alignment(
        phenotype_positions, sbs_positions,
        phenotype_pixel_size=phenotype_pixel_size,
        sbs_pixel_size=sbs_pixel_size,
        **kwargs
    )
    
    # Step 3: Analyze overlap with the result
    if not result.empty and len(result) > 0:
        alignment = result.iloc[0]
        overlap_analysis = analyze_overlap_with_correct_scale(
            phenotype_positions, sbs_positions, alignment
        )
        
        # Add overlap analysis to result
        result = result.copy()
        for key, value in overlap_analysis.items():
            result[f'overlap_{key}'] = value