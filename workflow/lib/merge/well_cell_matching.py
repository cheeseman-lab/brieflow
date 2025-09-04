"""Library for cell-to-cell matching between phenotype and SBS datasets.

This module provides functions for aligning and matching cells from phenotype and SBS
datasets using spatial transformations. It includes coordinate system alignment,
distance-based matching, and validation of match quality.

Key functions:
- filter_single_cell_tiles: Remove single-cell tiles that cause downstream issues
- load_alignment_parameters: Parse alignment parameters from various formats
- find_cell_matches: Main function for finding cell matches using spatial alignment
- validate_matches: Quality assessment of matching results
"""

import gc
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, Tuple


def filter_single_cell_tiles(
    matches_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter out single-cell tiles that cause phenotype extraction failures.

    Single-cell tiles are dropped during phenotype extraction due to early exit
    conditions in extract_phenotype_cp_multichannel, but they still appear in
    merge data. This causes NaN lookups in format_merge.

    Args:
        matches_df: DataFrame with cell matches containing plate, well, site, tile columns

    Returns:
        Tuple of (filtered_df, filtering_stats) where:
        - filtered_df: DataFrame with single-cell tiles removed
        - filtering_stats: Dictionary with filtering statistics
    """
    if matches_df.empty:
        return matches_df, {"tiles_removed": 0, "cells_removed": 0, "cells_kept": 0}

    # Count cells per tile
    tile_cell_counts = matches_df.groupby(["plate", "well", "site", "tile"]).size()

    # Identify single-cell tiles
    single_cell_tiles = tile_cell_counts[tile_cell_counts == 1].index

    if len(single_cell_tiles) == 0:
        return matches_df, {
            "tiles_removed": 0,
            "cells_removed": 0,
            "cells_kept": len(matches_df),
        }

    # Create mask for tiles to keep
    keep_mask = pd.Series(True, index=matches_df.index)

    for plate_id, well_id, site_id, tile_id in single_cell_tiles:
        tile_mask = (
            (matches_df["plate"] == plate_id)
            & (matches_df["well"] == well_id)
            & (matches_df["site"] == site_id)
            & (matches_df["tile"] == tile_id)
        )
        keep_mask &= ~tile_mask

    filtered_df = matches_df[keep_mask].copy()

    filtering_stats = {
        "tiles_removed": len(single_cell_tiles),
        "cells_removed": len(matches_df) - len(filtered_df),
        "cells_kept": len(filtered_df),
    }

    return filtered_df, filtering_stats


def load_alignment_parameters(alignment_row: pd.Series) -> Dict[str, Any]:
    """Load alignment parameters from a DataFrame row.

    Parses alignment parameters from various formats including string representations
    of arrays and handles different data types that may result from parquet storage.

    Args:
        alignment_row: Single row from alignment parameters DataFrame containing
                      rotation_matrix_flat, translation_vector, scale_factor, etc.

    Returns:
        Dictionary with parsed alignment parameters:
        - rotation: 2x2 numpy array for rotation transformation
        - translation: 2-element numpy array for translation
        - scale_factor: float for coordinate scaling
        - score: alignment quality score
        - determinant: transformation determinant
        - transformation_type: string describing transformation type
        - approach: string describing alignment approach
        - validation_mean_distance: mean distance from validation
    """
    # Extract rotation matrix - handle different formats
    rotation_flat = alignment_row.get("rotation_matrix_flat", [1.0, 0.0, 0.0, 1.0])

    if isinstance(rotation_flat, str):
        try:
            # Try direct evaluation for list format like "[1.0, 0.0, 0.0, 1.0]"
            rotation_flat = eval(rotation_flat)
        except SyntaxError:
            try:
                # Try numpy array string format like "[1. 0. 0. 1.]"
                clean_str = rotation_flat.strip("[]")
                rotation_flat = [float(x) for x in clean_str.split()]
            except (ValueError, AttributeError):
                try:
                    # Use numpy to parse array string
                    rotation_flat = np.fromstring(
                        rotation_flat.strip("[]"), sep=" "
                    ).tolist()
                except:
                    # Fallback: identity matrix
                    print(
                        f"Warning: Could not parse rotation matrix '{rotation_flat}', using identity"
                    )
                    rotation_flat = [1.0, 0.0, 0.0, 1.0]

    # Ensure it's a list and has 4 elements
    if not isinstance(rotation_flat, (list, tuple)) or len(rotation_flat) != 4:
        print(
            f"Warning: Invalid rotation matrix format, using identity. Got: {rotation_flat}"
        )
        rotation_flat = [1.0, 0.0, 0.0, 1.0]

    rotation = np.array(rotation_flat).reshape(2, 2)

    # Extract translation vector - handle different formats
    translation_list = alignment_row.get("translation_vector", [0.0, 0.0])

    # Handle pandas/parquet data types
    if isinstance(translation_list, np.ndarray):
        # NumPy array - convert to list
        translation_list = translation_list.tolist()
    elif isinstance(translation_list, str):
        # String representation - handle different formats
        
        # Check if it's a numpy array string format (has spaces between numbers)
        if " " in translation_list.strip("[]") and "," not in translation_list:
            # Numpy array format like "[-85.4958813  -95.79037779]"
            try:
                clean_str = translation_list.strip("[]")
                translation_list = [float(x) for x in clean_str.split()]
            except (ValueError, AttributeError):
                print(
                    f"Warning: Could not parse numpy array translation vector '{translation_list}', using zero"
                )
                translation_list = [0.0, 0.0]
        else:
            # Regular list format like "[-85.4, -95.8]"
            try:
                translation_list = eval(translation_list)
            except (SyntaxError, NameError):
                try:
                    clean_str = translation_list.strip("[]")
                    translation_list = [float(x) for x in clean_str.split()]
                except Exception:
                    print(
                        f"Warning: Could not parse translation vector '{translation_list}', using zero"
                    )
                    translation_list = [0.0, 0.0]

    # Ensure we have a proper list/tuple and handle different lengths
    if isinstance(translation_list, (list, tuple)):
        if len(translation_list) == 1:
            # Single element - assume this is X translation, Y is zero
            translation_list = [float(translation_list[0]), 0.0]
            print(
                f"Note: Using single-element translation vector: [{translation_list[0]:.1f}, 0.0]"
            )
        elif len(translation_list) == 2:
            # Two elements - use as is
            translation_list = [float(translation_list[0]), float(translation_list[1])]
        elif len(translation_list) == 0:
            # Empty - use zero
            print(f"Warning: Empty translation vector, using zero")
            translation_list = [0.0, 0.0]
        else:
            # Invalid length - truncate to first 2 elements or pad with zeros
            if len(translation_list) > 2:
                print(
                    f"Warning: Translation vector too long ({len(translation_list)}), using first 2 elements"
                )
                translation_list = [
                    float(translation_list[0]),
                    float(translation_list[1]),
                ]
            else:
                print(
                    f"Warning: Invalid translation vector length ({len(translation_list)}), using zero"
                )
                translation_list = [0.0, 0.0]
    else:
        # Not a list/tuple - check if it's a scalar that should be zero
        try:
            # Try to convert to float - might be a scalar
            scalar_val = float(translation_list)
            translation_list = [scalar_val, 0.0]
            print(
                f"Note: Converting scalar translation to vector: [{scalar_val:.1f}, 0.0]"
            )
        except (ValueError, TypeError):
            print(
                f"Warning: Invalid translation vector type ({type(translation_list)}), using zero. Got: {translation_list}"
            )
            translation_list = [0.0, 0.0]

    translation = np.array(translation_list)

    # Extract scale factor for coordinate system correction
    scale_factor = alignment_row.get("scale_factor", 1.0)
    if isinstance(scale_factor, str):
        try:
            scale_factor = float(scale_factor)
        except:
            print(f"Warning: Could not parse scale factor '{scale_factor}', using 1.0")
            scale_factor = 1.0

    return {
        "rotation": rotation,
        "translation": translation,
        "scale_factor": float(scale_factor),
        "score": float(alignment_row.get("score", 0.0)),
        "determinant": float(alignment_row.get("determinant", 1.0)),
        "transformation_type": str(alignment_row.get("transformation_type", "unknown")),
        "approach": str(alignment_row.get("approach", "unknown")),
        "validation_mean_distance": float(
            alignment_row.get("validation_mean_distance", 0.0)
        ),
    }


def find_cell_matches(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float = 10.0,
    chunk_size: int = 50000,
    transformed_phenotype_positions: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Find cell matches using alignment transformation.
    
    Matches cells between phenotype and SBS datasets using spatial alignment.
    Automatically chooses between direct and chunked approaches based on memory
    requirements.

    Args:
        phenotype_positions: DataFrame with phenotype cell positions (i, j columns)
        sbs_positions: DataFrame with SBS cell positions (i, j columns)
        alignment: Dictionary with alignment parameters (rotation, translation, scale_factor)
        threshold: Maximum distance for valid matches in pixels
        chunk_size: Size of chunks for memory-efficient processing
        transformed_phenotype_positions: Pre-calculated transformed coordinates (optional)

    Returns:
        Tuple of (matches_df, stats) where:
        - matches_df: DataFrame with matched cells and their properties
        - stats: Dictionary with matching statistics and performance metrics
    """
    print(
        f"Finding matches: {len(phenotype_positions):,} phenotype × {len(sbs_positions):,} SBS cells"
    )
    print(f"Distance threshold: {threshold}px")

    # Extract transformation parameters
    rotation = alignment.get("rotation", np.eye(2))
    translation = alignment.get("translation", np.zeros(2))
    scale_factor = alignment.get("scale_factor", 1.0)

    # Get coordinates for matching
    sbs_coords = sbs_positions[["i", "j"]].values

    # Use provided transformed coordinates OR calculate them
    if transformed_phenotype_positions is not None:
        print("Using pre-calculated transformed coordinates")
        transformed_coords = transformed_phenotype_positions[["i", "j"]].values
    else:
        print("Calculating transformed coordinates")
        pheno_coords = phenotype_positions[["i", "j"]].values
        transformed_coords = pheno_coords @ rotation.T + translation

    # Calculate memory requirement for full matrix
    total_comparisons = len(sbs_positions) * len(phenotype_positions)
    memory_required_gb = (total_comparisons * 8) / (1024**3)  # 8 bytes per float64

    # Choose approach based on memory requirements
    if memory_required_gb < 400:
        print(f"Using direct approach ({memory_required_gb:.1f}GB required)")
        raw_matches, stats = _find_matches_direct(
            phenotype_positions,
            sbs_positions,
            transformed_coords,
            sbs_coords,
            threshold,
            scale_factor,
        )
    else:
        print(
            f"Using chunked approach ({memory_required_gb:.1f}GB required, using chunks)"
        )
        raw_matches, stats = _find_matches_chunked(
            phenotype_positions,
            sbs_positions,
            transformed_coords,
            sbs_coords,
            threshold,
            chunk_size,
            scale_factor,
        )

    return raw_matches, stats


def _find_matches_direct(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float,
    scale_factor: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Direct approach optimized for large datasets with sufficient memory.
    
    Computes full distance matrix in memory for optimal performance when
    memory requirements are manageable.

    Args:
        phenotype_positions: DataFrame with phenotype cell data
        sbs_positions: DataFrame with SBS cell data  
        transformed_coords: Transformed phenotype coordinates
        sbs_coords: SBS coordinates
        threshold: Distance threshold for valid matches
        scale_factor: Scale factor for coordinate system

    Returns:
        Tuple of (matches_df, stats) with match results and statistics
    """
    print(
        f"Computing distance matrix: {len(sbs_coords):,} × {len(transformed_coords):,}"
    )

    # Calculate distances using coordinates in same coordinate system
    distances = cdist(sbs_coords, transformed_coords, metric="euclidean")

    # For each SBS cell, find closest phenotype cell
    closest_pheno_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)

    # Clear the large distance matrix immediately
    del distances
    gc.collect()

    # Filter by threshold
    valid_sbs_mask = min_distances < threshold
    n_valid = valid_sbs_mask.sum()

    print(f"Found {n_valid:,} matches within {threshold}px threshold")

    if n_valid == 0:
        return pd.DataFrame(), {"raw_matches": 0, "method": "direct"}

    # Get valid matches
    valid_sbs_indices = np.where(valid_sbs_mask)[0]
    valid_pheno_indices = closest_pheno_idx[valid_sbs_mask]
    valid_distances = min_distances[valid_sbs_mask]

    # Build matches DataFrame
    raw_matches = _build_matches_dataframe(
        phenotype_positions,
        sbs_positions,
        valid_pheno_indices,
        valid_sbs_indices,
        valid_distances,
        transformed_coords,
        sbs_coords,
        scale_factor,
    )

    stats = {
        "raw_matches": len(raw_matches),
        "method": "direct",
        "mean_distance": float(valid_distances.mean()),
        "max_distance": float(valid_distances.max()),
        "matches_within_threshold": n_valid,
        "threshold_used": threshold,
        "scale_factor_used": scale_factor,
    }

    print(f"Direct matching complete: {len(raw_matches):,} matches")

    return raw_matches, stats


def _find_matches_chunked(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float,
    chunk_size: int,
    scale_factor: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Chunked approach with aggressive memory management.
    
    Processes large datasets in chunks to avoid memory overflow while
    maintaining matching accuracy.

    Args:
        phenotype_positions: DataFrame with phenotype cell data
        sbs_positions: DataFrame with SBS cell data
        transformed_coords: Transformed phenotype coordinates
        sbs_coords: SBS coordinates  
        threshold: Distance threshold for valid matches
        chunk_size: Number of SBS cells to process per chunk
        scale_factor: Scale factor for coordinate system

    Returns:
        Tuple of (matches_df, stats) with match results and statistics
    """
    all_matches = []
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size

    print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks")

    total_matches = 0
    all_distances = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))
        chunk_size_actual = end_idx - start_idx

        if chunk_idx % 2 == 0:  # Print progress every other chunk
            print(
                f"Processing chunk {chunk_idx + 1}/{n_chunks} ({chunk_size_actual:,} cells)"
            )

        # Get chunk of SBS coordinates
        sbs_chunk_coords = sbs_coords[start_idx:end_idx]

        # Calculate distances for this chunk
        distances = cdist(sbs_chunk_coords, transformed_coords, metric="euclidean")

        # Find closest phenotype cell for each SBS cell in chunk
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)

        # Clear the distance matrix immediately
        del distances
        gc.collect()

        # Filter by threshold
        valid_matches = min_distances < threshold
        chunk_matches = valid_matches.sum()
        total_matches += chunk_matches

        if chunk_matches > 0:
            # Get indices for this chunk
            chunk_sbs_indices = np.arange(start_idx, end_idx)[valid_matches]
            chunk_pheno_indices = closest_pheno_idx[valid_matches]
            chunk_distances = min_distances[valid_matches]

            # Build chunk matches
            chunk_matches_df = _build_matches_dataframe(
                phenotype_positions,
                sbs_positions,
                chunk_pheno_indices,
                chunk_sbs_indices,
                chunk_distances,
                transformed_coords,
                sbs_coords,
                scale_factor,
            )

            all_matches.append(chunk_matches_df)
            all_distances.extend(chunk_distances.tolist())

        # Clear chunk variables
        del sbs_chunk_coords, closest_pheno_idx, min_distances, valid_matches
        if "chunk_distances" in locals():
            del chunk_distances, chunk_pheno_indices, chunk_sbs_indices
        gc.collect()

    print(f"Chunking complete: {total_matches:,} total matches found")

    # Combine all chunks
    if all_matches:
        raw_matches = pd.concat(all_matches, ignore_index=True)
        del all_matches
        gc.collect()
    else:
        raw_matches = pd.DataFrame()

    # Calculate stats
    if len(all_distances) > 0:
        mean_distance = float(np.mean(all_distances))
        max_distance = float(np.max(all_distances))
    else:
        mean_distance = 0.0
        max_distance = 0.0

    stats = {
        "raw_matches": len(raw_matches),
        "method": "chunked",
        "chunks_processed": n_chunks,
        "chunks_with_matches": len(all_matches)
        if "all_matches" in locals() and all_matches
        else 0,
        "chunk_size": chunk_size,
        "mean_distance": mean_distance,
        "max_distance": max_distance,
        "matches_within_threshold": total_matches,
        "threshold_used": threshold,
        "scale_factor_used": scale_factor,
    }

    print(f"Chunked matching complete: {len(raw_matches):,} matches")

    return raw_matches, stats


def _build_matches_dataframe(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    pheno_indices: np.ndarray,
    sbs_indices: np.ndarray,
    distances: np.ndarray,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    scale_factor: float,
) -> pd.DataFrame:
    """Build matches DataFrame using the coordinates that were used for distance calculation.
    
    Creates the final matches DataFrame with cell IDs, coordinates, and metadata.
    Uses 'stitched_cell_id' for coordinates matching when available.

    Args:
        phenotype_positions: DataFrame with phenotype cell data
        sbs_positions: DataFrame with SBS cell data
        pheno_indices: Array of phenotype cell indices for matches
        sbs_indices: Array of SBS cell indices for matches
        distances: Array of match distances
        transformed_coords: Transformed phenotype coordinates used for matching
        sbs_coords: SBS coordinates used for matching
        scale_factor: Scale factor applied during matching

    Returns:
        DataFrame with matched cell pairs and their properties
    """
    # Determine which cell ID column to use for each dataset
    # Priority: stitched_cell_id > label > original_cell_id
    pheno_id_col = None
    sbs_id_col = None

    for col_name in ["stitched_cell_id", "label", "original_cell_id"]:
        if pheno_id_col is None and col_name in phenotype_positions.columns:
            pheno_id_col = col_name
        if sbs_id_col is None and col_name in sbs_positions.columns:
            sbs_id_col = col_name

    if pheno_id_col is None:
        raise ValueError(
            "No suitable phenotype cell ID column found (stitched_cell_id, label, or original_cell_id)"
        )
    if sbs_id_col is None:
        raise ValueError(
            "No suitable SBS cell ID column found (stitched_cell_id, label, or original_cell_id)"
        )

    print(f"Using phenotype ID column: {pheno_id_col}")
    print(f"Using SBS ID column: {sbs_id_col}")

    matches_df = pd.DataFrame(
        {
            "cell_0": phenotype_positions.iloc[pheno_indices][pheno_id_col].values,
            "i_0": transformed_coords[pheno_indices, 0],
            "j_0": transformed_coords[pheno_indices, 1],
            "cell_1": sbs_positions.iloc[sbs_indices][sbs_id_col].values,
            "i_1": sbs_coords[sbs_indices, 0],
            "j_1": sbs_coords[sbs_indices, 1],
            "distance": distances,
        }
    )

    # Add area columns from original positions if available
    if "area" in phenotype_positions.columns:
        matches_df["area_0"] = phenotype_positions.iloc[pheno_indices]["area"].values
    else:
        matches_df["area_0"] = np.nan

    if "area" in sbs_positions.columns:
        matches_df["area_1"] = sbs_positions.iloc[sbs_indices]["area"].values
    else:
        matches_df["area_1"] = np.nan

    return matches_df


def validate_matches(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate cell matches and return quality metrics.

    Analyzes the quality of cell matches including distance statistics,
    duplication rates, and overall quality flags.

    Args:
        matches_df: DataFrame with cell matches containing distance and cell ID columns

    Returns:
        Dictionary with comprehensive validation metrics:
        - status: Overall validation status
        - match_count: Total number of matches
        - unique_phenotype_cells/unique_sbs_cells: Count of unique cells
        - distance_stats: Mean, median, max, std of match distances
        - distance_distribution: Counts at various distance thresholds
        - duplication: Analysis of duplicate matches
        - quality_flags: Overall quality assessment
    """
    if matches_df.empty:
        return {"status": "empty", "match_count": 0}

    # Basic counts
    n_matches = len(matches_df)
    n_unique_pheno = matches_df["cell_0"].nunique()
    n_unique_sbs = matches_df["cell_1"].nunique()

    # Distance statistics
    distances = matches_df["distance"]
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
    pheno_duplicates = matches_df["cell_0"].duplicated().sum()
    sbs_duplicates = matches_df["cell_1"].duplicated().sum()

    # Quality flags
    has_duplicates = pheno_duplicates > 0 or sbs_duplicates > 0
    has_large_distances = over_20px > 0
    good_quality = (
        not has_duplicates and mean_dist < 5.0 and over_20px < n_matches * 0.05
    )

    return {
        "status": "valid",
        "match_count": n_matches,
        "unique_phenotype_cells": n_unique_pheno,
        "unique_sbs_cells": n_unique_sbs,
        "distance_stats": {
            "mean": float(mean_dist),
            "median": float(median_dist),
            "max": float(max_dist),
            "std": float(std_dist),
        },
        "distance_distribution": {
            "under_1px": int(under_1px),
            "under_2px": int(under_2px),
            "under_5px": int(under_5px),
            "under_10px": int(under_10px),
            "over_20px": int(over_20px),
        },
        "duplication": {
            "phenotype_duplicates": int(pheno_duplicates),
            "sbs_duplicates": int(sbs_duplicates),
            "has_duplicates": has_duplicates,
        },
        "quality_flags": {
            "has_duplicates": has_duplicates,
            "has_large_distances": has_large_distances,
            "good_quality": good_quality,
        },
    }

def filter_tiles_by_diversity(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Filter out tiles that have only one unique original_cell_id.
    
    Removes tiles with insufficient cell diversity which can cause issues
    in downstream processing. Tiles with only one unique cell ID are typically
    artifacts or low-quality regions.

    Args:
        df: DataFrame with cell data containing 'tile' and 'original_cell_id' columns
        data_type: String describing the data type for logging (e.g., "Phenotype", "SBS")

    Returns:
        Filtered DataFrame containing only tiles with multiple unique cell IDs
    """
    if "original_cell_id" not in df.columns or "tile" not in df.columns:
        print(
            f"⚠️  WARNING: Cannot filter {data_type} tiles - missing original_cell_id or tile columns"
        )
        return df

    # Count unique original_cell_ids per tile
    tile_diversity = df.groupby("tile")["original_cell_id"].nunique()

    # Find tiles with more than 1 unique original_cell_id
    diverse_tiles = tile_diversity[tile_diversity > 1].index

    # Filter to keep only diverse tiles
    filtered_df = df[df["tile"].isin(diverse_tiles)]

    removed_tiles = len(tile_diversity) - len(diverse_tiles)
    removed_cells = len(df) - len(filtered_df)

    print(f"{data_type} tile diversity filtering:")
    print(f"  Input: {len(df):,} cells across {len(tile_diversity)} tiles")
    print(f"  Removed: {removed_tiles} tiles with single cell diversity")
    print(f"  Output: {len(filtered_df):,} cells across {len(diverse_tiles)} tiles")
    print(f"  Tiles kept: {sorted(diverse_tiles.tolist())}")

    return filtered_df

# Create empty output files when processing fails
def create_empty_outputs(reason: str) -> None:
    """Create empty output files when processing fails.
    
    Generates placeholder output files with proper structure when the main
    processing pipeline encounters errors. Ensures downstream steps can
    continue with empty results.

    Args:
        reason: String describing the reason for failure
    """
    empty_columns = [
        "plate",
        "well", 
        "site",
        "tile",
        "cell_0",
        "i_0",
        "j_0",
        "area_0",
        "cell_1",
        "i_1",
        "j_1",
        "area_1",
        "distance",
        "stitched_cell_id_0",
        "stitched_cell_id_1",
    ]

    empty_matches = pd.DataFrame(columns=empty_columns)

    # Add default values
    empty_matches["plate"] = snakemake.params.plate
    empty_matches["well"] = snakemake.params.well
    empty_matches["site"] = 1
    empty_matches["tile"] = 1

    # Save both outputs
    empty_matches.to_parquet(str(snakemake.output.raw_matches))
    empty_matches.to_parquet(str(snakemake.output.merged_cells))

    # Create failure summary as TSV (key-value format for aggregation script)
    summary_data = {
        "metric": [
            "status",
            "reason", 
            "plate",
            "well",
            "output_format_columns_included",
            "output_format_cell_0_contains",
            "output_format_cell_1_contains"
        ],
        "value": [
            "failed",
            reason,
            snakemake.params.plate,
            snakemake.params.well,
            ";".join(empty_columns),
            "original_phenotype_cell_ids",
            "original_sbs_cell_ids"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(str(snakemake.output.merge_summary), sep='\t', index=False)

    print(f"❌ Created empty outputs due to: {reason}")