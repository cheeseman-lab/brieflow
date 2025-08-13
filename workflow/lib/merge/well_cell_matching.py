"""
Step 2 Library: Cell-to-cell matching functions.
Saved this as: lib/merge/well_cell_matching.py
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, Tuple


def load_alignment_parameters(alignment_row: pd.Series) -> Dict[str, Any]:
    """
    Load alignment parameters from a DataFrame row.
    
    Args:
        alignment_row: Single row from alignment parameters DataFrame
        
    Returns:
        Dictionary with alignment parameters
    """
    # Extract rotation matrix - handle different formats
    rotation_flat = alignment_row.get('rotation_matrix_flat', [1.0, 0.0, 0.0, 1.0])
    
    if isinstance(rotation_flat, str):
        # Handle string representation - try multiple parsing methods
        try:
            # First try: direct eval (for list format like "[1.0, 0.0, 0.0, 1.0]")
            rotation_flat = eval(rotation_flat)
        except SyntaxError:
            try:
                # Second try: numpy array string format like "[1. 0. 0. 1.]"
                # Remove brackets and split by spaces, then convert to float
                clean_str = rotation_flat.strip('[]')
                rotation_flat = [float(x) for x in clean_str.split()]
            except (ValueError, AttributeError):
                try:
                    # Third try: use numpy to parse array string
                    import numpy as np
                    rotation_flat = np.fromstring(rotation_flat.strip('[]'), sep=' ').tolist()
                except:
                    # Fallback: identity matrix
                    print(f"Warning: Could not parse rotation matrix '{rotation_flat}', using identity")
                    rotation_flat = [1.0, 0.0, 0.0, 1.0]
    
    # Ensure it's a list and has 4 elements
    if not isinstance(rotation_flat, (list, tuple)) or len(rotation_flat) != 4:
        print(f"Warning: Invalid rotation matrix format, using identity. Got: {rotation_flat}")
        rotation_flat = [1.0, 0.0, 0.0, 1.0]
    
    rotation = np.array(rotation_flat).reshape(2, 2)
    
    # Extract translation vector - handle different formats
    translation_list = alignment_row.get('translation_vector', [0.0, 0.0])
    if isinstance(translation_list, str):
        try:
            # Try direct eval first
            translation_list = eval(translation_list)
        except SyntaxError:
            try:
                # Handle numpy array string format
                clean_str = translation_list.strip('[]')
                translation_list = [float(x) for x in clean_str.split()]
            except:
                try:
                    # Use numpy to parse
                    import numpy as np
                    translation_list = np.fromstring(translation_list.strip('[]'), sep=' ').tolist()
                except:
                    # Fallback: zero translation
                    print(f"Warning: Could not parse translation vector '{translation_list}', using zero")
                    translation_list = [0.0, 0.0]
    
    # Ensure it's a list and has 2 elements
    if not isinstance(translation_list, (list, tuple)) or len(translation_list) != 2:
        print(f"Warning: Invalid translation vector format, using zero. Got: {translation_list}")
        translation_list = [0.0, 0.0]
    
    translation = np.array(translation_list)
    
    # Debug output
    print(f"Loaded alignment parameters:")
    print(f"  Rotation matrix: {rotation}")
    print(f"  Translation: {translation}")
    print(f"  Determinant: {np.linalg.det(rotation):.6f}")
    
    return {
        'rotation': rotation,
        'translation': translation,
        'score': float(alignment_row.get('score', 0.0)),
        'determinant': float(alignment_row.get('determinant', 1.0)),
        'transformation_type': str(alignment_row.get('transformation_type', 'unknown')),
        'approach': str(alignment_row.get('approach', 'unknown')),
        'scale_factor': float(alignment_row.get('scale_factor', 1.0)),
        'validation_mean_distance': float(alignment_row.get('validation_mean_distance', 0.0))
    }

def find_cell_matches(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 10.0,
    chunk_size: int = 50000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find cell matches using alignment transformation.
    
    Args:
        phenotype_positions: Phenotype cell positions (should be pre-scaled)
        sbs_positions: SBS cell positions
        alignment: Alignment parameters dictionary
        threshold: Distance threshold for matches
        chunk_size: Chunk size for memory management
        
    Returns:
        Tuple of (raw_matches_df, summary_stats_dict)
    """
    print(f"Finding cell matches with threshold={threshold}px")
    
    # Extract transformation parameters
    rotation = alignment.get('rotation', np.eye(2))
    translation = alignment.get('translation', np.zeros(2))
    
    print(f"Using transformation: det={np.linalg.det(rotation):.6f}")
    
    # Get coordinates
    pheno_coords = phenotype_positions[['i', 'j']].values
    sbs_coords = sbs_positions[['i', 'j']].values
    
    # Transform phenotype coordinates to SBS coordinate system
    transformed_coords = pheno_coords @ rotation.T + translation
    
    print(f"Coordinate ranges after transformation:")
    print(f"  Transformed phenotype: i=[{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}]")
    print(f"  SBS: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")
    
    # Process in chunks or all at once based on size
    if len(sbs_positions) * len(phenotype_positions) < 1e9:  # Less than 1B calculations
        print(f"Using direct approach (small dataset)")
        raw_matches, stats = _find_matches_direct(
            phenotype_positions, sbs_positions, 
            transformed_coords, sbs_coords, threshold
        )
    else:
        print(f"Using chunked approach (large dataset)")
        raw_matches, stats = _find_matches_chunked(
            phenotype_positions, sbs_positions,
            transformed_coords, sbs_coords, threshold, chunk_size
        )
    
    return raw_matches, stats


def _find_matches_direct(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Direct approach for small datasets."""
    
    # Calculate all distances at once
    distances = cdist(sbs_coords, transformed_coords, metric='euclidean')
    
    # For each SBS cell, find closest phenotype cell
    closest_pheno_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    # Filter by threshold
    valid_sbs_mask = min_distances < threshold
    
    if valid_sbs_mask.sum() == 0:
        return pd.DataFrame(), {'raw_matches': 0, 'method': 'direct'}
    
    # Get valid matches
    valid_sbs_indices = np.where(valid_sbs_mask)[0]
    valid_pheno_indices = closest_pheno_idx[valid_sbs_mask]
    valid_distances = min_distances[valid_sbs_mask]
    
    # Build matches DataFrame
    raw_matches = _build_matches_dataframe(
        phenotype_positions, sbs_positions,
        valid_pheno_indices, valid_sbs_indices, valid_distances
    )
    
    stats = {
        'raw_matches': len(raw_matches),
        'method': 'direct',
        'mean_distance': float(valid_distances.mean()),
        'max_distance': float(valid_distances.max())
    }
    
    return raw_matches, stats


def _find_matches_chunked(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float,
    chunk_size: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Chunked approach for large datasets."""
    
    all_matches = []
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size
    
    print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks of {chunk_size:,}")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))
        
        if chunk_idx % 10 == 0:
            print(f"  Processing chunk {chunk_idx + 1}/{n_chunks}")
        
        # Get chunk of SBS coordinates
        sbs_chunk_coords = sbs_coords[start_idx:end_idx]
        
        # Calculate distances for this chunk
        distances = cdist(sbs_chunk_coords, transformed_coords, metric='euclidean')
        
        # Find closest phenotype cell for each SBS cell in chunk
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)
        
        # Filter by threshold
        valid_matches = min_distances < threshold
        
        if valid_matches.sum() > 0:
            # Get indices for this chunk
            chunk_sbs_indices = np.arange(start_idx, end_idx)[valid_matches]
            chunk_pheno_indices = closest_pheno_idx[valid_matches]
            chunk_distances = min_distances[valid_matches]
            
            # Build chunk matches
            chunk_matches = _build_matches_dataframe(
                phenotype_positions, sbs_positions,
                chunk_pheno_indices, chunk_sbs_indices, chunk_distances
            )
            
            all_matches.append(chunk_matches)
    
    # Combine all chunks
    if all_matches:
        raw_matches = pd.concat(all_matches, ignore_index=True)
    else:
        raw_matches = pd.DataFrame()
    
    stats = {
        'raw_matches': len(raw_matches),
        'method': 'chunked',
        'chunks_processed': n_chunks,
        'mean_distance': float(raw_matches['distance'].mean()) if len(raw_matches) > 0 else 0.0,
        'max_distance': float(raw_matches['distance'].max()) if len(raw_matches) > 0 else 0.0
    }
    
    return raw_matches, stats


def _build_matches_dataframe(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    pheno_indices: np.ndarray,
    sbs_indices: np.ndarray,
    distances: np.ndarray
) -> pd.DataFrame:
    """Build matches DataFrame from indices and distances."""
    
    matches_df = pd.DataFrame({
        'cell_0': phenotype_positions.iloc[pheno_indices]['cell'].values,
        'i_0': phenotype_positions.iloc[pheno_indices]['i'].values,
        'j_0': phenotype_positions.iloc[pheno_indices]['j'].values,
        'cell_1': sbs_positions.iloc[sbs_indices]['cell'].values,
        'i_1': sbs_positions.iloc[sbs_indices]['i'].values,
        'j_1': sbs_positions.iloc[sbs_indices]['j'].values,
        'distance': distances
    })
    
    # Add area columns if available
    if 'area' in phenotype_positions.columns:
        matches_df['area_0'] = phenotype_positions.iloc[pheno_indices]['area'].values
    else:
        matches_df['area_0'] = np.nan
        
    if 'area' in sbs_positions.columns:
        matches_df['area_1'] = sbs_positions.iloc[sbs_indices]['area'].values
    else:
        matches_df['area_1'] = np.nan
    
    return matches_df


def validate_matches(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate cell matches and return quality metrics.
    
    Args:
        matches_df: DataFrame with cell matches
        
    Returns:
        Dictionary with validation metrics
    """
    if matches_df.empty:
        return {'status': 'empty', 'match_count': 0}
    
    # Basic counts
    n_matches = len(matches_df)
    n_unique_pheno = matches_df['cell_0'].nunique()
    n_unique_sbs = matches_df['cell_1'].nunique()
    
    # Distance statistics
    distances = matches_df['distance']
    mean_dist = distances.mean()
    median_dist = distances.median()
    max_dist = distances.max()
    std_dist = distances.std()
    
    # Distance thresholds
    under_1px = (distances < 1).sum()
    under_2px = (distances < 2).sum()
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_20px = (distances > 20).sum()
    
    # Duplication analysis
    pheno_duplicates = matches_df['cell_0'].duplicated().sum()
    sbs_duplicates = matches_df['cell_1'].duplicated().sum()
    
    # Quality flags
    has_duplicates = pheno_duplicates > 0 or sbs_duplicates > 0
    has_large_distances = over_20px > 0
    good_quality = not has_duplicates and mean_dist < 5.0 and over_20px < n_matches * 0.05
    
    return {
        'status': 'valid',
        'match_count': n_matches,
        'unique_phenotype_cells': n_unique_pheno,
        'unique_sbs_cells': n_unique_sbs,
        'distance_stats': {
            'mean': float(mean_dist),
            'median': float(median_dist),
            'max': float(max_dist),
            'std': float(std_dist)
        },
        'distance_distribution': {
            'under_1px': int(under_1px),
            'under_2px': int(under_2px),
            'under_5px': int(under_5px),
            'under_10px': int(under_10px),
            'over_20px': int(over_20px)
        },
        'duplication': {
            'phenotype_duplicates': int(pheno_duplicates),
            'sbs_duplicates': int(sbs_duplicates),
            'has_duplicates': has_duplicates
        },
        'quality_flags': {
            'has_duplicates': has_duplicates,
            'has_large_distances': has_large_distances,
            'good_quality': good_quality
        }
    }