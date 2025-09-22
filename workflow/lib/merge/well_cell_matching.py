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
    """Filter out single-cell tiles that cause phenotype extraction failures."""
    if matches_df.empty:
        return matches_df, {"tiles_removed": 0, "cells_removed": 0, "cells_kept": 0}

    # Count cells per tile
    tile_cell_counts = matches_df.groupby(["plate", "well", "site", "tile"]).size()
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
    """Load alignment parameters from a DataFrame row."""
    # Extract rotation matrix - handle different formats
    rotation_flat = alignment_row.get("rotation_matrix_flat", [1.0, 0.0, 0.0, 1.0])

    if isinstance(rotation_flat, str):
        try:
            rotation_flat = eval(rotation_flat)
        except SyntaxError:
            try:
                clean_str = rotation_flat.strip("[]")
                rotation_flat = [float(x) for x in clean_str.split()]
            except (ValueError, AttributeError):
                try:
                    rotation_flat = np.fromstring(
                        rotation_flat.strip("[]"), sep=" "
                    ).tolist()
                except:
                    print(
                        f"Warning: Could not parse rotation matrix '{rotation_flat}', using identity"
                    )
                    rotation_flat = [1.0, 0.0, 0.0, 1.0]

    if not isinstance(rotation_flat, (list, tuple)) or len(rotation_flat) != 4:
        print(
            f"Warning: Invalid rotation matrix format, using identity. Got: {rotation_flat}"
        )
        rotation_flat = [1.0, 0.0, 0.0, 1.0]

    rotation = np.array(rotation_flat).reshape(2, 2)

    # Extract translation vector - handle different formats
    translation_list = alignment_row.get("translation_vector", [0.0, 0.0])

    if isinstance(translation_list, np.ndarray):
        translation_list = translation_list.tolist()
    elif isinstance(translation_list, str):
        if " " in translation_list.strip("[]") and "," not in translation_list:
            try:
                clean_str = translation_list.strip("[]")
                translation_list = [float(x) for x in clean_str.split()]
            except (ValueError, AttributeError):
                print(
                    f"Warning: Could not parse numpy array translation vector '{translation_list}', using zero"
                )
                translation_list = [0.0, 0.0]
        else:
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

    if isinstance(translation_list, (list, tuple)):
        if len(translation_list) == 1:
            translation_list = [float(translation_list[0]), 0.0]
            print(
                f"Note: Using single-element translation vector: [{translation_list[0]:.1f}, 0.0]"
            )
        elif len(translation_list) == 2:
            translation_list = [float(translation_list[0]), float(translation_list[1])]
        elif len(translation_list) == 0:
            print(f"Warning: Empty translation vector, using zero")
            translation_list = [0.0, 0.0]
        else:
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
        try:
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

    # Extract scale factor
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
    chunk_size: int = 10000,  # Reduced from 50000 to 10000
    transformed_phenotype_positions: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Find cell matches using alignment transformation."""
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
    memory_required_gb = (total_comparisons * 8) / (1024**3)

    # Use more conservative memory threshold to force chunking for large datasets
    if memory_required_gb < 50:  # Reduced from 400 to 50
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
    """Direct approach optimized for large datasets with sufficient memory."""
    print(
        f"Computing distance matrix: {len(sbs_coords):,} × {len(transformed_coords):,}"
    )

    distances = cdist(sbs_coords, transformed_coords, metric="euclidean")
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
    """Chunked approach with aggressive memory management and clean logging."""
    all_matches = []
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size

    print(
        f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks of {chunk_size:,}"
    )

    total_matches = 0
    all_distances = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))
        chunk_size_actual = end_idx - start_idx

        # Clean progress reporting - show every 10% or every 5 chunks, whichever is less frequent
        progress_interval = max(1, min(5, n_chunks // 10))
        if chunk_idx % progress_interval == 0:
            progress = (chunk_idx / n_chunks) * 100
            print(
                f"  Progress: {progress:.0f}% (chunk {chunk_idx + 1}/{n_chunks}, {chunk_size_actual:,} cells)"
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
    """Build matches DataFrame using the coordinates that were used for distance calculation."""
    # Determine which cell ID column to use for each dataset
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

    print(f"Using phenotype ID column: {pheno_id_col}, SBS ID column: {sbs_id_col}")

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
    """Validate cell matches and return quality metrics."""
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
    """Filter out tiles that have only one unique original_cell_id with clean logging."""
    if "original_cell_id" not in df.columns or "tile" not in df.columns:
        print(f"⚠️  Cannot filter {data_type} tiles - missing required columns")
        return df

    # Count unique original_cell_ids per tile
    tile_diversity = df.groupby("tile")["original_cell_id"].nunique()
    diverse_tiles = tile_diversity[tile_diversity > 1].index
    filtered_df = df[df["tile"].isin(diverse_tiles)]

    removed_tiles = len(tile_diversity) - len(diverse_tiles)

    print(
        f"{data_type} filtering: {len(df):,} → {len(filtered_df):,} cells "
        f"({removed_tiles} tiles removed)"
    )

    # Only show tile details if reasonable number, otherwise show summary
    if len(diverse_tiles) <= 20:
        print(f"  Tiles kept: {sorted(diverse_tiles.tolist())}")
    else:
        tile_range = f"{min(diverse_tiles)}-{max(diverse_tiles)}"
        print(f"  Tiles kept: {len(diverse_tiles)} tiles (range: {tile_range})")

    return filtered_df


def create_empty_outputs(reason: str) -> None:
    """Create empty output files when processing fails."""
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

    # Create failure summary
    summary_data = {
        "metric": [
            "status",
            "reason",
            "plate",
            "well",
            "output_format_columns_included",
            "output_format_cell_0_contains",
            "output_format_cell_1_contains",
        ],
        "value": [
            "failed",
            reason,
            snakemake.params.plate,
            snakemake.params.well,
            ";".join(empty_columns),
            "original_phenotype_cell_ids",
            "original_sbs_cell_ids",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(str(snakemake.output.merge_summary), sep="\t", index=False)

    print(f"❌ Created empty outputs due to: {reason}")

def build_final_matches(
    raw_matches: pd.DataFrame,
    phenotype_filtered: pd.DataFrame,
    sbs_filtered: pd.DataFrame,
    plate: str,
    well: str,
) -> pd.DataFrame:
    """Build final matches DataFrame with metadata extraction.
    
    Args:
        raw_matches: Raw match results from find_cell_matches
        phenotype_filtered: Filtered phenotype positions
        sbs_filtered: Filtered SBS positions  
        plate: Plate identifier
        well: Well identifier
        
    Returns:
        DataFrame with complete match information including metadata
    """
    if raw_matches.empty:
        return _create_empty_matches_df(plate, well)
    
    print(f"Building final matches from {len(raw_matches):,} raw matches...")
    
    # Map stitched IDs back to row indices
    pheno_stitched_to_idx = (
        phenotype_filtered.reset_index()
        .set_index("stitched_cell_id")["index"]
        .to_dict()
    )
    sbs_stitched_to_idx = (
        sbs_filtered.reset_index()
        .set_index("stitched_cell_id")["index"]
        .to_dict()
    )
    
    # Get indices for matched cells
    pheno_indices = [pheno_stitched_to_idx[cell_id] for cell_id in raw_matches["cell_0"]]
    sbs_indices = [sbs_stitched_to_idx[cell_id] for cell_id in raw_matches["cell_1"]]
    
    # Extract matched rows
    phenotype_match_rows = phenotype_filtered.loc[pheno_indices]
    sbs_match_rows = sbs_filtered.loc[sbs_indices]
    
    # Build final DataFrame
    final_matches = pd.DataFrame({
        "plate": plate,
        "well": well,
        "site": sbs_match_rows["tile"].values,
        "tile": phenotype_match_rows["tile"].values,
        "cell_0": phenotype_match_rows["original_cell_id"].values,
        "cell_1": sbs_match_rows["original_cell_id"].values,
        "i_0": raw_matches["i_0"].values,
        "j_0": raw_matches["j_0"].values,
        "i_1": raw_matches["i_1"].values,
        "j_1": raw_matches["j_1"].values,
        "area_0": phenotype_match_rows.get("area", pd.Series([np.nan] * len(pheno_indices))).values,
        "area_1": sbs_match_rows.get("area", pd.Series([np.nan] * len(sbs_indices))).values,
        "distance": raw_matches["distance"].values,
        "stitched_cell_id_0": phenotype_match_rows["stitched_cell_id"].values,
        "stitched_cell_id_1": sbs_match_rows["stitched_cell_id"].values,
    })
    
    final_matches["site"] = final_matches["site"].astype(int)
    final_matches["tile"] = final_matches["tile"].astype(int)
    
    print(f"✅ Built {len(final_matches):,} final matches")
    
    return final_matches


def create_merge_summary(
    final_matches: pd.DataFrame,
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    phenotype_filtered: pd.DataFrame,
    sbs_filtered: pd.DataFrame,
    alignment: dict,
    threshold: float,
    plate: str,
    well: str,
) -> pd.DataFrame:
    """Create comprehensive merge summary with validation metrics.
    
    Args:
        final_matches: Final matched cells DataFrame
        phenotype_scaled: Original phenotype positions before filtering
        sbs_positions: Original SBS positions before filtering
        phenotype_filtered: Filtered phenotype positions
        sbs_filtered: Filtered SBS positions
        alignment: Alignment parameters dictionary
        threshold: Distance threshold used for matching
        plate: Plate identifier
        well: Well identifier
        
    Returns:
        Single-row DataFrame with comprehensive statistics
    """
    if final_matches.empty:
        return _create_failure_summary(
            plate, well, threshold, phenotype_scaled, sbs_positions, "no_matches_found"
        )
    
    # Validate and log quality
    validation_results = validate_matches(final_matches)
    _log_validation_results(validation_results)
    
    # Build summary dictionary
    summary_dict = {
        "plate": plate,
        "well": well,
        "status": "success",
        "distance_threshold_pixels": float(threshold),
        "phenotype_cells_before_filtering": len(phenotype_scaled),
        "sbs_cells_before_filtering": len(sbs_positions),
        "phenotype_cells_after_filtering": len(phenotype_filtered),
        "sbs_cells_after_filtering": len(sbs_filtered),
        "phenotype_tiles_removed": _count_tiles_removed(phenotype_scaled, phenotype_filtered),
        "sbs_tiles_removed": _count_tiles_removed(sbs_positions, sbs_filtered),
        "alignment_approach": str(alignment.get("approach", "unknown")),
        "alignment_transformation_type": str(alignment.get("transformation_type", "unknown")),
        "alignment_score": float(alignment.get("score", 0)),
        "alignment_determinant": float(alignment.get("determinant", 1)),
        "raw_matches_found": len(final_matches),
        "mean_match_distance": float(final_matches["distance"].mean()),
        "max_match_distance": float(final_matches["distance"].max()),
        "matches_under_5px": int((final_matches["distance"] < 5).sum()),
        "matches_under_10px": int((final_matches["distance"] < 10).sum()),
        "match_rate_phenotype": float(len(final_matches) / len(phenotype_filtered)) if len(phenotype_filtered) > 0 else 0.0,
        "match_rate_sbs": float(len(final_matches) / len(sbs_filtered)) if len(sbs_filtered) > 0 else 0.0,
    }
    
    # Add validation metrics
    if validation_results:
        summary_dict.update(_extract_validation_metrics(validation_results))
    
    return pd.DataFrame([summary_dict])


def _create_empty_matches_df(plate: str, well: str) -> pd.DataFrame:
    """Create empty matches DataFrame with correct schema."""
    return pd.DataFrame(
        columns=[
            "plate", "well", "site", "tile",
            "cell_0", "cell_1", "i_0", "j_0", "i_1", "j_1",
            "area_0", "area_1", "distance",
            "stitched_cell_id_0", "stitched_cell_id_1",
        ]
    ).assign(plate=plate, well=well)


def _create_failure_summary(
    plate: str, well: str, threshold: float,
    phenotype_scaled: pd.DataFrame, sbs_positions: pd.DataFrame,
    reason: str
) -> pd.DataFrame:
    """Create failure summary for empty outputs."""
    return pd.DataFrame([{
        "plate": plate,
        "well": well,
        "status": "failed",
        "failure_reason": reason,
        "distance_threshold_pixels": float(threshold),
        "phenotype_cells_before_filtering": len(phenotype_scaled),
        "sbs_cells_before_filtering": len(sbs_positions),
        "phenotype_cells_after_filtering": 0,
        "sbs_cells_after_filtering": 0,
        "phenotype_tiles_removed": 0,
        "sbs_tiles_removed": 0,
        "alignment_approach": "unknown",
        "alignment_transformation_type": "unknown",
        "alignment_score": 0.0,
        "alignment_determinant": 1.0,
        "raw_matches_found": 0,
        "mean_match_distance": 0.0,
        "max_match_distance": 0.0,
        "matches_under_5px": 0,
        "matches_under_10px": 0,
        "match_rate_phenotype": 0.0,
        "match_rate_sbs": 0.0,
        "validation_status": "failed",
    }])


def _count_tiles_removed(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> int:
    """Count number of tiles removed during filtering."""
    if "tile" not in original_df.columns:
        return 0
    return len(original_df["tile"].unique()) - len(filtered_df["tile"].unique())


def _log_validation_results(validation_results: dict) -> None:
    """Log validation results with appropriate warnings."""
    if validation_results.get("status") != "valid":
        print(f"❌ WARNING: Match validation failed: {validation_results.get('status', 'unknown')}")
        return
    
    # Quality assessment
    if validation_results.get("quality_flags", {}).get("good_quality", False):
        print("✅ Match quality: GOOD")
    else:
        print("⚠️  Match quality: ACCEPTABLE")
    
    # Distance stats
    dist_stats = validation_results.get("distance_stats", {})
    dist_dist = validation_results.get("distance_distribution", {})
    print(f"Distance: mean={dist_stats.get('mean', 0):.1f}px, max={dist_stats.get('max', 0):.1f}px")
    print(f"Quality: {dist_dist.get('under_5px', 0)} under 5px, {dist_dist.get('under_10px', 0)} under 10px")
    
    # Warnings
    if validation_results.get("duplication", {}).get("has_duplicates", False):
        dups = validation_results["duplication"]
        print(f"⚠️  WARNING: Duplicates (phenotype: {dups['phenotype_duplicates']}, SBS: {dups['sbs_duplicates']})")
    
    if dist_dist.get("over_20px", 0) > 0:
        print(f"⚠️  WARNING: {dist_dist['over_20px']} matches have distance >20px")


def _extract_validation_metrics(validation_results: dict) -> dict:
    """Extract validation metrics into flat dictionary for summary."""
    metrics = {
        "validation_status": validation_results.get("status", "unknown"),
    }
    
    # Distance stats
    for key, value in validation_results.get("distance_stats", {}).items():
        metrics[f"validation_distance_{key}"] = value
    
    # Distance distribution
    for key, value in validation_results.get("distance_distribution", {}).items():
        metrics[f"validation_distribution_{key}"] = value
    
    # Quality flags
    for key, value in validation_results.get("quality_flags", {}).items():
        metrics[f"validation_quality_{key}"] = value
    
    # Duplication info
    for key, value in validation_results.get("duplication", {}).items():
        metrics[f"validation_duplication_{key}"] = value
    
    return metrics