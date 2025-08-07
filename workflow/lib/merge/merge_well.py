"""
Memory-efficient additions for merge_well.py
Add these functions to your existing merge_well.py file.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings
import gc
from typing import Tuple

# Import existing functions (these should already be imported in your file)
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


def memory_efficient_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
    max_cells_for_hash: int = 50000,
    triangle_distance_threshold: float = 0.3,
    min_matching_triangles: int = 10
) -> pd.DataFrame:
    """
    Memory-efficient alignment between phenotype and SBS using subsampled triangle hashing.
    """
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        print("Empty position data, cannot perform alignment")
        return pd.DataFrame()
    
    print(f"Starting memory-efficient alignment:")
    print(f"  Phenotype cells: {len(phenotype_positions)}")
    print(f"  SBS cells: {len(sbs_positions)}")
    print(f"  Max cells for hashing: {max_cells_for_hash}")
    
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
        print(f"\n--- Triangle matching ---")
        print(f"Phenotype triangles: {len(phenotype_hash)}")
        print(f"SBS triangles: {len(sbs_hash)}")
        
        # Extract vectors and centers
        V_0, c_0 = get_vc(phenotype_hash)
        V_1, c_1 = get_vc(sbs_hash)
        
        # Use chunked nearest neighbors to avoid memory issues
        chunk_size = min(10000, len(V_0))
        i0, i1, distances = chunked_nearest_neighbors(V_0, V_1, chunk_size)
        
        # Filter based on distance threshold
        filt = distances < triangle_distance_threshold
        n_matching = filt.sum()
        
        print(f"Triangles within distance threshold: {n_matching}")
        
        if n_matching < min_matching_triangles:
            print(f"Only {n_matching} matching triangles found, need at least {min_matching_triangles}")
            return pd.DataFrame()
        
        # Get matching triangle centers
        X, Y = c_0[i0[filt]], c_1[i1[filt]]
        
        print(f"Using {len(X)} matching triangle centers for transformation")
        
        # Fit transformation using RANSAC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(
                min_samples=min(len(X), 10),
                max_trials=1000,
                residual_threshold=5.0
            )
            model.fit(X, Y)
        
        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_
        
        # Score the transformation more efficiently
        print("--- Scoring transformation ---")
        
        # Use a subset of centers for scoring to avoid memory issues
        max_centers_for_scoring = 5000
        if len(c_0) > max_centers_for_scoring:
            score_indices = np.random.choice(len(c_0), max_centers_for_scoring, replace=False)
            c_0_score = c_0[score_indices]
        else:
            c_0_score = c_0
        
        predicted_centers = model.predict(c_0_score)
        
        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(c_1)
        distances_score, _ = tree.query(predicted_centers)
        
        threshold_region = 50
        filt_score = distances_score < threshold_region
        score = (distances_score[filt_score] < 2).mean() if filt_score.sum() > 0 else 0
        
        # Calculate determinant
        determinant = np.linalg.det(rotation)
        
        # Create result
        result = pd.DataFrame([{
            'rotation_1': rotation[0] if len(rotation) > 0 else [0, 0],
            'rotation_2': rotation[1] if len(rotation) > 1 else [0, 0], 
            'translation': translation,
            'score': score,
            'determinant': determinant,
            'well': phenotype_positions['well'].iloc[0] if len(phenotype_positions) > 0 else 'unknown',
            'n_triangles_matched': n_matching,
            'n_triangles_phenotype': len(c_0),
            'n_triangles_sbs': len(c_1),
            'cells_used_phenotype': len(phenotype_hash),
            'cells_used_sbs': len(sbs_hash)
        }])
        
        print(f"✅ Alignment results:")
        print(f"   Score: {score:.3f}")
        print(f"   Determinant: {determinant:.3f}")
        print(f"   Matched triangles: {n_matching}")
        print(f"   Cells used: {len(phenotype_hash)} phenotype, {len(sbs_hash)} SBS")
        
        return result
        
    except Exception as e:
        print(f"Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def memory_efficient_merge_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame, 
    alignment: pd.Series,
    threshold: float = 2.0,
    chunk_size: int = 50000
) -> pd.DataFrame:
    """
    Memory-efficient cell merging using chunked processing.
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
    

def stitched_well_alignment(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    det_range: Tuple[float, float] = (0.8, 1.2),
    score_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Perform alignment between phenotype and SBS stitched wells using actual cell positions.
    
    This function automatically detects if memory-efficient processing is needed.

    Args:
        phenotype_positions: Cell positions in phenotype stitched well
        sbs_positions: Cell positions in SBS stitched well
        det_range: Acceptable range for transformation determinant
        score_threshold: Minimum score for valid alignment

    Returns:
        DataFrame with alignment parameters
    """
    # Automatically use memory-efficient approach for large datasets
    total_cells = len(phenotype_positions) + len(sbs_positions)
    memory_threshold = 100000  # Use memory-efficient approach if > 100k total cells
    
    if total_cells > memory_threshold:
        print(f"Large dataset detected ({total_cells} total cells), using memory-efficient approach")
        return memory_efficient_well_alignment(
            phenotype_positions=phenotype_positions,
            sbs_positions=sbs_positions,
            det_range=det_range,
            score_threshold=score_threshold,
            max_cells_for_hash=50000,  # Conservative limit
            triangle_distance_threshold=0.3,
            min_matching_triangles=10
        )
    
    # Original approach for smaller datasets (your existing code below)
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        print("Empty position data, cannot perform alignment")
        return pd.DataFrame()

    # Generate triangle hashes for both datasets
    phenotype_hash = hash_stitched_cell_positions(phenotype_positions)
    sbs_hash = hash_stitched_cell_positions(sbs_positions)

    if len(phenotype_hash) == 0 or len(sbs_hash) == 0:
        print("Empty hash data, cannot perform alignment")
        return pd.DataFrame()

    try:
        # Extract vectors and centers
        V_0, c_0 = get_vc(phenotype_hash)
        V_1, c_1 = get_vc(sbs_hash)

        # Find nearest neighbors between triangle vectors
        i0, i1, distances = nearest_neighbors(V_0, V_1)

        # Filter based on distance threshold
        triangle_threshold = 0.3  # Triangle matching threshold
        filt = distances < triangle_threshold
        if filt.sum() < 5:
            print(f"Only {filt.sum()} matching triangles found, need at least 5")
            return pd.DataFrame()

        # Get matching triangle centers
        X, Y = c_0[i0[filt]], c_1[i1[filt]]

        # Fit transformation using RANSAC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = RANSACRegressor(min_samples=5, max_trials=1000)
            model.fit(X, Y)

        rotation = model.estimator_.coef_
        translation = model.estimator_.intercept_

        # Score the transformation
        predicted_centers = model.predict(c_0)
        distances_score = cdist(predicted_centers, c_1, metric="sqeuclidean")
        threshold_region = 50
        filt_score = np.sqrt(distances_score.min(axis=0)) < threshold_region
        score = (np.sqrt(distances_score.min(axis=0))[filt_score] < 2).mean()

        # Calculate determinant
        determinant = np.linalg.det(rotation)

        # Create result
        result = pd.DataFrame(
            [
                {
                    "rotation_1": rotation[0] if len(rotation) > 0 else [0, 0],
                    "rotation_2": rotation[1] if len(rotation) > 1 else [0, 0],
                    "translation": translation,
                    "score": score,
                    "determinant": determinant,
                    "well": phenotype_positions["well"].iloc[0]
                    if len(phenotype_positions) > 0
                    else "unknown",
                    "n_triangles_matched": filt.sum(),
                    "n_triangles_phenotype": len(c_0),
                    "n_triangles_sbs": len(c_1),
                }
            ]
        )

        print(
            f"Alignment results: score={score:.3f}, determinant={determinant:.3f}, matched_triangles={filt.sum()}"
        )

        return result

    except Exception as e:
        print(f"Alignment failed: {e}")
        return pd.DataFrame()


def merge_stitched_cells(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: pd.Series,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Merge cell positions between phenotype and SBS stitched wells.
    
    This function automatically detects if memory-efficient processing is needed.

    Args:
        phenotype_positions: Cell positions in phenotype well
        sbs_positions: Cell positions in SBS well
        alignment: Alignment parameters (rotation, translation)
        threshold: Maximum distance for cell matching

    Returns:
        DataFrame with merged cell identities
    """
    # Automatically use memory-efficient approach for large datasets
    total_cells = len(phenotype_positions) + len(sbs_positions)
    memory_threshold = 100000  # Use memory-efficient approach if > 100k total cells
    
    if total_cells > memory_threshold:
        print(f"Large dataset detected ({total_cells} total cells), using memory-efficient merge")
        return memory_efficient_merge_cells(
            phenotype_positions=phenotype_positions,
            sbs_positions=sbs_positions,
            alignment=alignment,
            threshold=threshold,
            chunk_size=50000  # Conservative chunk size
        )
    
    # Original approach for smaller datasets (your existing code below)
    if len(phenotype_positions) == 0 or len(sbs_positions) == 0:
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "cell_0",
                "i_0",
                "j_0",
                "area_0",
                "cell_1",
                "i_1",
                "j_1",
                "area_1",
                "distance",
            ]
        )

    try:
        # Build transformation model
        rotation = np.array([alignment["rotation_1"], alignment["rotation_2"]])
        translation = alignment["translation"]
        model = LinearRegression()
        model.coef_ = rotation
        model.intercept_ = translation

        # Extract coordinates
        X = phenotype_positions[["i", "j"]].values
        Y = sbs_positions[["i", "j"]].values

        # Apply transformation to phenotype coordinates
        X_transformed = model.predict(X)

        # Calculate distances between transformed phenotype and SBS
        distances = cdist(X_transformed, Y, metric="euclidean")

        # Find nearest neighbors
        ix = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)

        # Filter by threshold
        filt = min_distances < threshold

        if filt.sum() == 0:
            print("No matches found within threshold")
            return pd.DataFrame(
                columns=[
                    "plate",
                    "well",
                    "cell_0",
                    "i_0",
                    "j_0",
                    "area_0",
                    "cell_1",
                    "i_1",
                    "j_1",
                    "area_1",
                    "distance",
                ]
            )

        # Create merged dataframe
        matched_phenotype = phenotype_positions[filt].reset_index(drop=True)
        matched_sbs = sbs_positions.iloc[ix[filt]].reset_index(drop=True)

        merged_data = pd.DataFrame(
            {
                "plate": 1,  # You may need to extract this from your data
                "well": matched_phenotype["well"],
                "cell_0": matched_phenotype["cell"],
                "i_0": matched_phenotype["i"],
                "j_0": matched_phenotype["j"],
                "area_0": matched_phenotype["area"],
                "cell_1": matched_sbs["cell"],
                "i_1": matched_sbs["i"],
                "j_1": matched_sbs["j"],
                "area_1": matched_sbs["area"],
                "distance": min_distances[filt],
            }
        )

        print(f"Successfully merged {len(merged_data)} cells (threshold={threshold})")
        return merged_data

    except Exception as e:
        print(f"Merge failed: {e}")
        return pd.DataFrame(
            columns=[
                "plate",
                "well",
                "cell_0",
                "i_0",
                "j_0",
                "area_0",
                "cell_1",
                "i_1",
                "j_1",
                "area_1",
                "distance",
            ]
        )