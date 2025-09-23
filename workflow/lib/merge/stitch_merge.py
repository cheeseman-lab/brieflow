"""Library for cell-to-cell matching between phenotype and SBS datasets.

This module provides functions for aligning and matching cells from phenotype and SBS
datasets using spatial transformations. It includes coordinate system alignment,
distance-based matching, and validation of match quality.
"""

import gc
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, Tuple, Optional


# Schema definitions
MATCHES_COLUMNS = [
    "plate", "well", "site", "tile", "cell_0", "cell_1",
    "i_0", "j_0", "i_1", "j_1", "area_0", "area_1",
    "distance", "stitched_cell_id_0", "stitched_cell_id_1"
]


def load_alignment_parameters(alignment_row: pd.Series) -> Dict[str, Any]:
    """Load alignment parameters from a DataFrame row.

    Single validation check for data integrity.
    """
    # Extract rotation matrix
    rotation_flat = alignment_row.get("rotation_matrix_flat", [1.0, 0.0, 0.0, 1.0])

    if isinstance(rotation_flat, str):
        rotation_flat = eval(rotation_flat)

    if (
        not isinstance(rotation_flat, (list, tuple, np.ndarray))
        or len(rotation_flat) != 4
    ):
        print(f"Warning: Invalid rotation matrix, using identity: {rotation_flat}")
        rotation_flat = [1.0, 0.0, 0.0, 1.0]

    rotation = np.array(rotation_flat).reshape(2, 2)

    # Extract translation vector
    translation_list = alignment_row.get("translation_vector", [0.0, 0.0])

    if isinstance(translation_list, str):
        translation_list = eval(translation_list)
    elif isinstance(translation_list, np.ndarray):
        translation_list = translation_list.tolist()

    if not isinstance(translation_list, (list, tuple)) or len(translation_list) != 2:
        print(f"Warning: Invalid translation vector, using zero: {translation_list}")
        translation_list = [0.0, 0.0]

    translation = np.array(translation_list, dtype=float)

    # Extract scale factor
    scale_factor = float(alignment_row.get("scale_factor", 1.0))

    return {
        "rotation": rotation,
        "translation": translation,
        "scale_factor": scale_factor,
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
    chunk_size: int = 10000,
    transformed_phenotype_positions: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Find cell matches using alignment transformation.
    
    Args:
        phenotype_positions: Phenotype cell positions
        sbs_positions: SBS cell positions
        alignment: Alignment parameters dictionary
        threshold: Distance threshold for matching (pixels)
        chunk_size: Chunk size for memory-efficient processing
        transformed_phenotype_positions: Pre-calculated transformed coordinates
        verbose: Print progress messages
    
    Returns:
        Tuple of (matches DataFrame, statistics dictionary)
    """
    if verbose:
        print(
            f"Finding matches: {len(phenotype_positions):,} phenotype × {len(sbs_positions):,} SBS cells"
        )
        print(f"Distance threshold: {threshold}px")

    # Get coordinates
    sbs_coords = sbs_positions[["i", "j"]].values

    if transformed_phenotype_positions is not None:
        if verbose:
            print("Using pre-calculated transformed coordinates")
        transformed_coords = transformed_phenotype_positions[["i", "j"]].values
    else:
        if verbose:
            print("Calculating transformed coordinates")
        rotation = alignment.get("rotation", np.eye(2))
        translation = alignment.get("translation", np.zeros(2))
        pheno_coords = phenotype_positions[["i", "j"]].values
        transformed_coords = pheno_coords @ rotation.T + translation

    # Determine processing approach based on memory requirements
    total_comparisons = len(sbs_positions) * len(phenotype_positions)
    memory_required_gb = (total_comparisons * 8) / (1024**3)

    if memory_required_gb < 50:
        if verbose:
            print(f"Using direct approach ({memory_required_gb:.1f}GB required)")
        return _find_matches_direct(
            phenotype_positions,
            sbs_positions,
            transformed_coords,
            sbs_coords,
            threshold,
            alignment.get("scale_factor", 1.0),
            verbose,
        )
    else:
        if verbose:
            print(f"Using chunked approach ({memory_required_gb:.1f}GB required)")
        return _find_matches_chunked(
            phenotype_positions,
            sbs_positions,
            transformed_coords,
            sbs_coords,
            threshold,
            chunk_size,
            alignment.get("scale_factor", 1.0),
            verbose,
        )


def _process_distance_chunk(
    sbs_coords: np.ndarray,
    transformed_coords: np.ndarray,
    threshold: float,
    sbs_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distances and filter by threshold for a coordinate chunk.
    
    Returns:
        Tuple of (valid_sbs_indices, valid_pheno_indices, valid_distances)
    """
    # Calculate distances
    distances = cdist(sbs_coords, transformed_coords, metric="euclidean")
    closest_pheno_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)

    del distances
    gc.collect()

    # Filter by threshold
    valid_mask = min_distances < threshold
    valid_sbs_idx = np.arange(len(sbs_coords))[valid_mask] + sbs_offset
    valid_pheno_idx = closest_pheno_idx[valid_mask]
    valid_distances = min_distances[valid_mask]

    return valid_sbs_idx, valid_pheno_idx, valid_distances


def _find_matches_direct(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float,
    scale_factor: float,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Direct distance calculation approach."""
    if verbose:
        print(
            f"Computing distance matrix: {len(sbs_coords):,} × {len(transformed_coords):,}"
        )

    # Process all coordinates at once
    valid_sbs_idx, valid_pheno_idx, valid_distances = _process_distance_chunk(
        sbs_coords, transformed_coords, threshold
    )

    n_valid = len(valid_distances)

    if verbose:
        print(f"Found {n_valid:,} matches within {threshold}px threshold")

    if n_valid == 0:
        return pd.DataFrame(), {"raw_matches": 0, "method": "direct"}

    # Build matches
    matches_df = _build_matches_dataframe(
        phenotype_positions,
        sbs_positions,
        valid_pheno_idx,
        valid_sbs_idx,
        valid_distances,
        transformed_coords,
        sbs_coords,
    )

    stats = {
        "raw_matches": len(matches_df),
        "method": "direct",
        "mean_distance": float(valid_distances.mean()),
        "max_distance": float(valid_distances.max()),
        "matches_within_threshold": n_valid,
        "threshold_used": threshold,
        "scale_factor_used": scale_factor,
    }

    return matches_df, stats


def _find_matches_chunked(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
    threshold: float,
    chunk_size: int,
    scale_factor: float,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Chunked distance calculation for memory efficiency."""
    n_chunks = (len(sbs_positions) + chunk_size - 1) // chunk_size

    if verbose:
        print(f"Processing {len(sbs_positions):,} SBS cells in {n_chunks} chunks")

    all_matches = []
    all_distances = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_positions))

        if verbose and chunk_idx % max(1, n_chunks // 10) == 0:
            progress = (chunk_idx / n_chunks) * 100
            print(f"  Progress: {progress:.0f}% (chunk {chunk_idx + 1}/{n_chunks})")

        # Calculate distances for chunk
        sbs_chunk = sbs_coords[start_idx:end_idx]
        chunk_sbs_idx, chunk_pheno_idx, chunk_distances = _process_distance_chunk(
            sbs_chunk, transformed_coords, threshold, sbs_offset=start_idx
        )

        if len(chunk_distances) > 0:
            chunk_matches = _build_matches_dataframe(
                phenotype_positions,
                sbs_positions,
                chunk_pheno_idx,
                chunk_sbs_idx,
                chunk_distances,
                transformed_coords,
                sbs_coords,
            )

            all_matches.append(chunk_matches)
            all_distances.extend(chunk_distances.tolist())

    # Combine results
    if all_matches:
        raw_matches = pd.concat(all_matches, ignore_index=True)
    else:
        raw_matches = pd.DataFrame()

    stats = {
        "raw_matches": len(raw_matches),
        "method": "chunked",
        "chunks_processed": n_chunks,
        "chunk_size": chunk_size,
        "mean_distance": float(np.mean(all_distances)) if all_distances else 0.0,
        "max_distance": float(np.max(all_distances)) if all_distances else 0.0,
        "matches_within_threshold": len(all_distances),
        "threshold_used": threshold,
        "scale_factor_used": scale_factor,
    }

    if verbose:
        print(f"Chunking complete: {len(raw_matches):,} total matches")

    return raw_matches, stats


def _build_matches_dataframe(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    pheno_indices: np.ndarray,
    sbs_indices: np.ndarray,
    distances: np.ndarray,
    transformed_coords: np.ndarray,
    sbs_coords: np.ndarray,
) -> pd.DataFrame:
    """Build matches DataFrame from indices and distances.

    Uses stitched_cell_id as the primary identifier for both datasets.
    """
    matches_df = pd.DataFrame(
        {
            "cell_0": phenotype_positions.iloc[pheno_indices][
                "stitched_cell_id"
            ].values,
            "i_0": transformed_coords[pheno_indices, 0],
            "j_0": transformed_coords[pheno_indices, 1],
            "cell_1": sbs_positions.iloc[sbs_indices]["stitched_cell_id"].values,
            "i_1": sbs_coords[sbs_indices, 0],
            "j_1": sbs_coords[sbs_indices, 1],
            "distance": distances,
        }
    )

    # Add area columns if available
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

    # Distance statistics
    distances = matches_df["distance"]

    # Distance thresholds
    under_1px = (distances < 1).sum()
    under_2px = (distances < 2).sum()
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_20px = (distances > 20).sum()

    # Duplication analysis
    pheno_duplicates = matches_df["cell_0"].duplicated().sum()
    sbs_duplicates = matches_df["cell_1"].duplicated().sum()

    return {
        "status": "valid",
        "match_count": n_matches,
        "unique_phenotype_cells": matches_df["cell_0"].nunique(),
        "unique_sbs_cells": matches_df["cell_1"].nunique(),
        "distance_stats": {
            "mean": float(distances.mean()),
            "median": float(distances.median()),
            "max": float(distances.max()),
            "std": float(distances.std()),
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
            "has_duplicates": pheno_duplicates > 0 or sbs_duplicates > 0,
        },
    }


def filter_tiles_by_diversity(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Filter out tiles that have only one unique original_cell_id."""
    if "original_cell_id" not in df.columns or "tile" not in df.columns:
        print(f"Cannot filter {data_type} tiles - missing required columns")
        return df

    # Count unique original_cell_ids per tile
    tile_diversity = df.groupby("tile")["original_cell_id"].nunique()
    diverse_tiles = tile_diversity[tile_diversity > 1].index
    filtered_df = df[df["tile"].isin(diverse_tiles)]

    removed_tiles = len(tile_diversity) - len(diverse_tiles)

    print(
        f"{data_type} filtering: {len(df):,} → {len(filtered_df):,} cells ({removed_tiles} tiles removed)"
    )

    if len(diverse_tiles) <= 20:
        print(f"  Tiles kept: {sorted(diverse_tiles.tolist())}")
    else:
        tile_range = f"{min(diverse_tiles)}-{max(diverse_tiles)}"
        print(f"  Tiles kept: {len(diverse_tiles)} tiles (range: {tile_range})")

    return filtered_df


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
    pheno_id_map = (
        phenotype_filtered.reset_index()
        .set_index("stitched_cell_id")["index"]
        .to_dict()
    )
    sbs_id_map = (
        sbs_filtered.reset_index().set_index("stitched_cell_id")["index"].to_dict()
    )

    # Get indices for matched cells
    pheno_indices = [pheno_id_map[cell_id] for cell_id in raw_matches["cell_0"]]
    sbs_indices = [sbs_id_map[cell_id] for cell_id in raw_matches["cell_1"]]

    # Extract matched rows
    phenotype_rows = phenotype_filtered.loc[pheno_indices]
    sbs_rows = sbs_filtered.loc[sbs_indices]

    # Build final DataFrame
    final_matches = pd.DataFrame(
        {
            "plate": plate,
            "well": well,
            "site": sbs_rows["tile"].values,
            "tile": phenotype_rows["tile"].values,
            "cell_0": phenotype_rows["original_cell_id"].values,
            "cell_1": sbs_rows["original_cell_id"].values,
            "i_0": raw_matches["i_0"].values,
            "j_0": raw_matches["j_0"].values,
            "i_1": raw_matches["i_1"].values,
            "j_1": raw_matches["j_1"].values,
            "area_0": phenotype_rows.get(
                "area", pd.Series([np.nan] * len(pheno_indices))
            ).values,
            "area_1": sbs_rows.get(
                "area", pd.Series([np.nan] * len(sbs_indices))
            ).values,
            "distance": raw_matches["distance"].values,
            "stitched_cell_id_0": phenotype_rows["stitched_cell_id"].values,
            "stitched_cell_id_1": sbs_rows["stitched_cell_id"].values,
        }
    )

    print(f"Built {len(final_matches):,} final matches")

    return final_matches


def create_merge_summary(
    final_matches: pd.DataFrame,
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    phenotype_filtered: pd.DataFrame,
    sbs_filtered: pd.DataFrame,
    alignment: Dict[str, Any],
    threshold: float,
    plate: str,
    well: str,
) -> pd.DataFrame:
    """Create comprehensive merge summary with validation metrics."""
    if final_matches.empty:
        return _create_failure_summary(
            plate, well, threshold, phenotype_scaled, sbs_positions
        )

    # Validate and log quality
    validation = validate_matches(final_matches)
    _log_validation_results(validation)

    # Build summary dictionary
    summary = {
        "plate": plate,
        "well": well,
        "status": "success",
        "distance_threshold_pixels": threshold,
        "phenotype_cells_before_filtering": len(phenotype_scaled),
        "sbs_cells_before_filtering": len(sbs_positions),
        "phenotype_cells_after_filtering": len(phenotype_filtered),
        "sbs_cells_after_filtering": len(sbs_filtered),
        "phenotype_tiles_removed": _count_tiles_removed(
            phenotype_scaled, phenotype_filtered
        ),
        "sbs_tiles_removed": _count_tiles_removed(sbs_positions, sbs_filtered),
        "alignment_approach": alignment.get("approach", "unknown"),
        "alignment_transformation_type": alignment.get("transformation_type", "unknown"),
        "alignment_score": alignment.get("score", 0),
        "alignment_determinant": alignment.get("determinant", 1),
        "raw_matches_found": len(final_matches),
        "mean_match_distance": final_matches["distance"].mean(),
        "max_match_distance": final_matches["distance"].max(),
        "matches_under_5px": (final_matches["distance"] < 5).sum(),
        "matches_under_10px": (final_matches["distance"] < 10).sum(),
        "match_rate_phenotype": len(final_matches) / len(phenotype_filtered)
        if len(phenotype_filtered) > 0
        else 0.0,
        "match_rate_sbs": len(final_matches) / len(sbs_filtered)
        if len(sbs_filtered) > 0
        else 0.0,
    }

    # Add validation metrics
    summary.update(
        {
            "validation_status": validation.get("status", "unknown"),
            **{
                f"validation_distance_{k}": v
                for k, v in validation.get("distance_stats", {}).items()
            },
            **{
                f"validation_distribution_{k}": v
                for k, v in validation.get("distance_distribution", {}).items()
            },
            **{
                f"validation_duplication_{k}": v
                for k, v in validation.get("duplication", {}).items()
            },
        }
    )

    return pd.DataFrame([summary])


def _create_empty_matches_df(plate: str, well: str) -> pd.DataFrame:
    """Create empty matches DataFrame with correct schema."""
    return pd.DataFrame(columns=MATCHES_COLUMNS).assign(plate=plate, well=well)


def _create_failure_summary(
    plate: str,
    well: str,
    threshold: float,
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Create failure summary for empty outputs."""
    return pd.DataFrame(
        [
            {
                "plate": plate,
                "well": well,
                "status": "failed",
                "failure_reason": "no_matches_found",
                "distance_threshold_pixels": threshold,
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
            }
        ]
    )


def _count_tiles_removed(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> int:
    """Count number of tiles removed during filtering."""
    if "tile" not in original_df.columns:
        return 0
    return len(original_df["tile"].unique()) - len(filtered_df["tile"].unique())


def _log_validation_results(validation: Dict[str, Any]) -> None:
    """Log validation results with warnings for issues."""
    if validation.get("status") != "valid":
        print(f"Match validation failed: {validation.get('status', 'unknown')}")
        return

    stats = validation.get("distance_stats", {})
    dist = validation.get("distance_distribution", {})
    dups = validation.get("duplication", {})

    # Distance stats
    print(
        f"Distance: mean={stats.get('mean', 0):.1f}px, max={stats.get('max', 0):.1f}px"
    )
    print(
        f"Quality: {dist.get('under_5px', 0)} under 5px, {dist.get('under_10px', 0)} under 10px"
    )

    # Warnings
    if dups.get("has_duplicates"):
        print(
            f"WARNING: Duplicates (phenotype: {dups['phenotype_duplicates']}, SBS: {dups['sbs_duplicates']})"
        )

    if dist.get("over_20px", 0) > 0:
        print(f"WARNING: {dist['over_20px']} matches >20px")