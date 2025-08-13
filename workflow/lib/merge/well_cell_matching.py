"""
Step 2 Library: Cell-to-cell matching functions.
FIXED VERSION: Coordinate system alignment corrected
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
    
    # FIXED: Extract scale factor for coordinate system correction
    scale_factor = alignment_row.get('scale_factor', 1.0)
    if isinstance(scale_factor, str):
        try:
            scale_factor = float(scale_factor)
        except:
            print(f"Warning: Could not parse scale factor '{scale_factor}', using 1.0")
            scale_factor = 1.0
    
    # Debug output
    print(f"Loaded alignment parameters:")
    print(f"  Rotation matrix: {rotation}")
    print(f"  Translation: {translation}")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Determinant: {np.linalg.det(rotation):.6f}")
    
    return {
        'rotation': rotation,
        'translation': translation,
        'scale_factor': float(scale_factor),  # FIXED: Include scale factor
        'score': float(alignment_row.get('score', 0.0)),
        'determinant': float(alignment_row.get('determinant', 1.0)),
        'transformation_type': str(alignment_row.get('transformation_type', 'unknown')),
        'approach': str(alignment_row.get('approach', 'unknown')),
        'validation_mean_distance': float(alignment_row.get('validation_mean_distance', 0.0))
    }

def find_cell_matches(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 10.0,
    chunk_size: int = 50000,
    transformed_phenotype_positions: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find cell matches using alignment transformation - FIXED coordinate system alignment.
    """
    print(f"Finding cell matches with threshold={threshold}px")
    print(f"Dataset sizes: {len(phenotype_positions):,} phenotype, {len(sbs_positions):,} SBS")
    
    # Extract transformation parameters
    rotation = alignment.get('rotation', np.eye(2))
    translation = alignment.get('translation', np.zeros(2))
    scale_factor = alignment.get('scale_factor', 1.0)  # FIXED: Get scale factor
    
    print(f"Using transformation:")
    print(f"  Rotation det: {np.linalg.det(rotation):.6f}")
    print(f"  Translation: {translation}")
    print(f"  Scale factor: {scale_factor}")
    
    # =================================================================
    # COORDINATE SYSTEM: Both should already be in same scale after alignment
    # =================================================================
    
    print(f"\n📐 COORDINATE SYSTEM: Using coordinates as-is (both already in same scale)")
    
    # Get SBS coordinates as-is (they should already be in the right scale)
    sbs_coords = sbs_positions[['i', 'j']].values
    
    print(f"Coordinate ranges:")
    print(f"  SBS range: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")
    
    # Use provided transformed coordinates OR calculate them
    if transformed_phenotype_positions is not None:
        print("Using pre-calculated transformed coordinates")
        transformed_coords = transformed_phenotype_positions[['i', 'j']].values
        print(f"Transformed phenotype range: i=[{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}]")
    else:
        print("Calculating transformed coordinates on-the-fly")
        # Get scaled phenotype coordinates and transform them
        pheno_coords = phenotype_positions[['i', 'j']].values
        transformed_coords = pheno_coords @ rotation.T + translation
        print(f"Calculated transformed range: i=[{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}]")
    
    # Now both coordinate systems should be in the same scale
    print(f"\n✅ Both coordinate systems already aligned - using as-is for distance calculation")
    
    # Calculate memory requirement for full matrix
    total_comparisons = len(sbs_positions) * len(phenotype_positions)
    memory_required_gb = (total_comparisons * 8) / (1024**3)  # 8 bytes per float64
    print(f"Full distance matrix would require: {memory_required_gb:.1f}GB")
    print(f"Available memory budget: 900GB")
    
    # Use more conservative memory threshold
    if memory_required_gb < 400: 
        print(f"Using direct approach (sufficient memory available)")
        raw_matches, stats = _find_matches_direct(
            phenotype_positions, sbs_positions, 
            transformed_coords, sbs_coords, threshold, scale_factor  # Use sbs_coords as-is
        )
    else:
        print(f"Using chunked approach (memory conservation)")
        raw_matches, stats = _find_matches_chunked(
            phenotype_positions, sbs_positions,
            transformed_coords, sbs_coords, threshold, chunk_size, scale_factor  # Use sbs_coords as-is
        )
    
    return raw_matches, stats


def _find_matches_direct(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,  # FIXED: Use SBS coordinates as-is
    threshold: float,
    scale_factor: float  # Keep for coordinate output handling
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Direct approach optimized for large datasets with sufficient memory.
    FIXED: Uses properly scaled coordinates.
    """
    import gc
    
    print(f"Calculating distance matrix: {len(sbs_coords):,} × {len(transformed_coords):,}")
    
    # Calculate distances using coordinates as-is (both already in same coordinate system)
    distances = cdist(sbs_coords, transformed_coords, metric='euclidean')
    
    print(f"Distance matrix calculated: {distances.shape}, {distances.nbytes / (1024**3):.1f}GB")
    
    # For each SBS cell, find closest phenotype cell
    print("Finding closest matches...")
    closest_pheno_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    # Clear the large distance matrix immediately
    del distances
    gc.collect()
    print("Distance matrix cleared from memory")
    
    # Filter by threshold
    valid_sbs_mask = min_distances < threshold
    n_valid = valid_sbs_mask.sum()
    
    print(f"Found {n_valid:,} matches within {threshold}px threshold")
    
    if n_valid == 0:
        return pd.DataFrame(), {'raw_matches': 0, 'method': 'direct'}
    
    # Get valid matches
    valid_sbs_indices = np.where(valid_sbs_mask)[0]
    valid_pheno_indices = closest_pheno_idx[valid_sbs_mask]
    valid_distances = min_distances[valid_sbs_mask]
    
    print(f"Building matches DataFrame with {len(valid_sbs_indices):,} matches...")
    
    # FIXED: Build matches DataFrame with original SBS coordinates
    raw_matches = _build_matches_dataframe(
        phenotype_positions, sbs_positions,
        valid_pheno_indices, valid_sbs_indices, valid_distances,
        transformed_coords, scale_factor  # FIXED: Pass scale factor for coordinate conversion
    )
    
    stats = {
        'raw_matches': len(raw_matches),
        'method': 'direct',
        'mean_distance': float(valid_distances.mean()),
        'max_distance': float(valid_distances.max()),
        'matches_within_threshold': n_valid,
        'threshold_used': threshold,
        'scale_factor_used': scale_factor  # FIXED: Include scale factor in stats
    }
    
    print(f"✅ Direct matching complete: {len(raw_matches):,} raw matches")
    
    return raw_matches, stats


def _find_matches_chunked(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,  # FIXED: Use SBS coordinates as-is
    threshold: float,
    chunk_size: int,
    scale_factor: float  # Keep for coordinate output handling
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Chunked approach with aggressive memory management.
    FIXED: Uses properly scaled coordinates.
    """
    import gc
    
    all_matches = []
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size
    
    print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks of {chunk_size:,}")
    print(f"Transformed phenotype coordinates: {len(transformed_coords):,} cells")
    
    total_matches = 0
    all_distances = []  # Track distances for final stats
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))
        chunk_size_actual = end_idx - start_idx
        
        print(f"Processing chunk {chunk_idx + 1}/{n_chunks} ({chunk_size_actual:,} cells, indices {start_idx}-{end_idx-1})")
        
        # Get chunk of SBS coordinates as-is
        sbs_chunk_coords = sbs_coords[start_idx:end_idx]
        
        # Calculate memory requirement for this chunk
        chunk_memory_gb = (chunk_size_actual * len(transformed_coords) * 8) / (1024**3)
        print(f"  Chunk memory requirement: {chunk_memory_gb:.1f}GB")
        
        # Calculate distances for this chunk
        print(f"  Calculating distances: {len(sbs_chunk_coords):,} × {len(transformed_coords):,}")
        distances = cdist(sbs_chunk_coords, transformed_coords, metric='euclidean')
        
        print(f"  Distance matrix shape: {distances.shape}")
        print(f"  Distance range: [{distances.min():.2f}, {distances.max():.2f}]")
        
        # Find closest phenotype cell for each SBS cell in chunk
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)
        
        print(f"  Min distances range: [{min_distances.min():.2f}, {min_distances.max():.2f}]")
        
        # Clear the distance matrix immediately
        del distances
        gc.collect()
        
        # Filter by threshold
        valid_matches = min_distances < threshold
        chunk_matches = valid_matches.sum()
        total_matches += chunk_matches
        
        print(f"  Found {chunk_matches:,} matches within {threshold}px threshold")
        
        if chunk_matches > 0:
            # Get indices for this chunk
            chunk_sbs_indices = np.arange(start_idx, end_idx)[valid_matches]
            chunk_pheno_indices = closest_pheno_idx[valid_matches]
            chunk_distances = min_distances[valid_matches]
            
            print(f"  Building DataFrame for {len(chunk_sbs_indices):,} matches")
            
            # FIXED: Build chunk matches with proper coordinate handling
            chunk_matches_df = _build_matches_dataframe(
                phenotype_positions, sbs_positions,
                chunk_pheno_indices, chunk_sbs_indices, chunk_distances,
                transformed_coords, scale_factor  # FIXED: Pass scale factor
            )
            
            all_matches.append(chunk_matches_df)
            all_distances.extend(chunk_distances.tolist())
            
            print(f"  ✅ Chunk {chunk_idx + 1} complete: {len(chunk_matches_df):,} matches added")
        else:
            print(f"  ⚠️ Chunk {chunk_idx + 1}: No matches found")
        
        # Clear chunk variables
        del sbs_chunk_coords, closest_pheno_idx, min_distances, valid_matches
        if 'chunk_distances' in locals():
            del chunk_distances
        if 'chunk_pheno_indices' in locals():
            del chunk_pheno_indices
        if 'chunk_sbs_indices' in locals():
            del chunk_sbs_indices
        gc.collect()
        
        print(f"  Running total: {total_matches:,} matches across {chunk_idx + 1} chunks")
    
    print(f"\n📊 Chunking Summary:")
    print(f"  Total chunks processed: {n_chunks}")
    print(f"  Total matches found: {total_matches:,}")
    print(f"  Non-empty chunks: {len(all_matches)}")
    
    # Combine all chunks
    if all_matches:
        print(f"Combining {len(all_matches)} chunk results...")
        raw_matches = pd.concat(all_matches, ignore_index=True)
        del all_matches  # Free memory
        gc.collect()
        
        print(f"✅ Combined result: {len(raw_matches):,} total matches")
    else:
        print("❌ No matches found in any chunk")
        raw_matches = pd.DataFrame()
    
    # Calculate stats
    if len(all_distances) > 0:
        mean_distance = float(np.mean(all_distances))
        max_distance = float(np.max(all_distances))
    else:
        mean_distance = 0.0
        max_distance = 0.0
    
    stats = {
        'raw_matches': len(raw_matches),
        'method': 'chunked',
        'chunks_processed': n_chunks,
        'chunks_with_matches': len(all_matches) if 'all_matches' in locals() and all_matches else 0,
        'chunk_size': chunk_size,
        'mean_distance': mean_distance,
        'max_distance': max_distance,
        'matches_within_threshold': total_matches,
        'threshold_used': threshold,
        'scale_factor_used': scale_factor  # FIXED: Include scale factor in stats
    }
    
    print(f"✅ Chunked matching complete: {len(raw_matches):,} raw matches")
    
    return raw_matches, stats


def _build_matches_dataframe(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    pheno_indices: np.ndarray,
    sbs_indices: np.ndarray,
    distances: np.ndarray,
    transformed_coords: np.ndarray,  # Transformed coordinates (already in right space)
    scale_factor: float  # Keep for reference but don't use for scaling
) -> pd.DataFrame:
    """
    Build matches DataFrame using coordinates as-is (both already in same coordinate system).
    """
    
    matches_df = pd.DataFrame({
        'cell_0': phenotype_positions.iloc[pheno_indices]['cell'].values,
        'i_0': transformed_coords[pheno_indices, 0],  # Transformed coordinates
        'j_0': transformed_coords[pheno_indices, 1],  # Transformed coordinates
        'cell_1': sbs_positions.iloc[sbs_indices]['cell'].values,
        'i_1': sbs_positions.iloc[sbs_indices]['i'].values,  # SBS coordinates as-is
        'j_1': sbs_positions.iloc[sbs_indices]['j'].values,  # SBS coordinates as-is
        'distance': distances
    })
    
    # Add area columns from ORIGINAL positions (areas don't change with coordinate scaling)
    if 'area' in phenotype_positions.columns:
        matches_df['area_0'] = phenotype_positions.iloc[pheno_indices]['area'].values
    else:
        matches_df['area_0'] = np.nan
        
    if 'area' in sbs_positions.columns:
        matches_df['area_1'] = sbs_positions.iloc[sbs_indices]['area'].values
    else:
        matches_df['area_1'] = np.nan
    
    print(f"    ✅ DataFrame built successfully: {len(matches_df):,} rows")
    print(f"    📐 Coordinates stored as-is (both datasets already in same coordinate system)")
    print(f"    📊 Manual distance calculation should now match stored distances")
    
    return matches_df


# Add debugging function for coordinate analysis
def debug_coordinate_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Debug coordinate alignment by checking transformation quality.
    """
    print(f"\n🔍 DEBUGGING COORDINATE ALIGNMENT")
    
    # Sample coordinates for analysis
    n_pheno_sample = min(sample_size, len(phenotype_positions))
    n_sbs_sample = min(sample_size, len(sbs_positions))
    
    pheno_sample = phenotype_positions.sample(n_pheno_sample, random_state=42)
    sbs_sample = sbs_positions.sample(n_sbs_sample, random_state=42)
    
    # Get transformation
    rotation = alignment.get('rotation', np.eye(2))
    translation = alignment.get('translation', np.zeros(2))
    
    # Transform phenotype coordinates
    pheno_coords = pheno_sample[['i', 'j']].values
    transformed_coords = pheno_coords @ rotation.T + translation
    
    # Calculate distances to all SBS coordinates
    sbs_coords = sbs_sample[['i', 'j']].values
    distances = cdist(transformed_coords, sbs_coords, metric='euclidean')
    
    # Find closest matches
    closest_sbs_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    debug_info = {
        'sample_sizes': {
            'phenotype': n_pheno_sample,
            'sbs': n_sbs_sample
        },
        'transformation': {
            'rotation_det': float(np.linalg.det(rotation)),
            'translation_norm': float(np.linalg.norm(translation))
        },
        'coordinate_ranges': {
            'original_phenotype': {
                'i': [float(pheno_coords[:, 0].min()), float(pheno_coords[:, 0].max())],
                'j': [float(pheno_coords[:, 1].min()), float(pheno_coords[:, 1].max())]
            },
            'transformed_phenotype': {
                'i': [float(transformed_coords[:, 0].min()), float(transformed_coords[:, 0].max())],
                'j': [float(transformed_coords[:, 1].min()), float(transformed_coords[:, 1].max())]
            },
            'sbs': {
                'i': [float(sbs_coords[:, 0].min()), float(sbs_coords[:, 0].max())],
                'j': [float(sbs_coords[:, 1].min()), float(sbs_coords[:, 1].max())]
            }
        },
        'distance_analysis': {
            'mean_closest_distance': float(min_distances.mean()),
            'median_closest_distance': float(np.median(min_distances)),
            'min_distance': float(min_distances.min()),
            'max_distance': float(min_distances.max()),
            'std_distance': float(min_distances.std()),
            'distances_under_10px': int((min_distances < 10).sum()),
            'distances_under_5px': int((min_distances < 5).sum()),
            'distances_under_2px': int((min_distances < 2).sum())
        }
    }
    
    print(f"Sample analysis results:")
    print(f"  Mean closest distance: {debug_info['distance_analysis']['mean_closest_distance']:.2f}px")
    print(f"  Matches under 10px: {debug_info['distance_analysis']['distances_under_10px']}/{n_pheno_sample}")
    print(f"  Matches under 5px: {debug_info['distance_analysis']['distances_under_5px']}/{n_pheno_sample}")
    
    return debug_info


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