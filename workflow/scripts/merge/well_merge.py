"""Enhanced well merge pipeline with separate output files for each step.

This script performs well-level cell merging between phenotype and SBS datasets
using a multi-step approach:
1. Coordinate scaling to align pixel sizes
2. Triangle hash generation for both datasets 
3. Triangle hash-based alignment or identity fallback
4. Fine cell-level alignment with distance filtering

The pipeline saves intermediate results at each step as separate Snakemake outputs.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import (
    well_level_triangle_hash,
    pure_scaling_alignment,
    merge_stitched_cells,
    triangle_hash_well_alignment,
)


def main():
    """Main execution function for the well merge pipeline."""
    print("=== ENHANCED WELL MERGE PIPELINE ===")

    # Load stitched cell positions
    phenotype_positions = validate_dtypes(
        pd.read_parquet(snakemake.input.phenotype_positions)
    )
    sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

    # Extract parameters
    plate = snakemake.params.plate
    well = snakemake.params.well
    scale_factor = snakemake.params.scale_factor
    threshold = snakemake.params.threshold

    print(f"Processing Plate {plate}, Well {well}")
    print(f"Phenotype cells: {len(phenotype_positions):,}")
    print(f"SBS cells: {len(sbs_positions):,}")
    print(f"Scale factor: {scale_factor}")
    print(f"Distance threshold: {threshold} px")

    # Step 1: Coordinate Scaling
    print("\n=== STEP 1: COORDINATE SCALING ===")
    
    phenotype_scaled = coordinate_scaling_step(
        phenotype_positions, scale_factor
    )
    
    overlap_fraction = validate_coordinate_overlap(
        phenotype_scaled, sbs_positions
    )

    # Save scaled phenotype positions
    phenotype_scaled.to_parquet(str(snakemake.output.phenotype_scaled))
    print(f"Saved scaled phenotype positions: {snakemake.output.phenotype_scaled}")

    # Step 2: Triangle Hashing
    print("\n=== STEP 2: TRIANGLE HASHING ===")
    
    phenotype_triangles, sbs_triangles = triangle_hashing_step(
        phenotype_scaled, sbs_positions
    )

    if len(phenotype_triangles) == 0 or len(sbs_triangles) == 0:
        print("Triangle hash generation failed - creating empty outputs")
        create_empty_outputs(
            snakemake.output, phenotype_scaled, scale_factor, plate, well
        )
        return

    # Save triangle hashes
    phenotype_triangles.to_parquet(str(snakemake.output.phenotype_triangles))
    sbs_triangles.to_parquet(str(snakemake.output.sbs_triangles))
    
    print(f"Saved phenotype triangles: {snakemake.output.phenotype_triangles}")
    print(f"Saved SBS triangles: {snakemake.output.sbs_triangles}")

    # Step 3: Alignment
    print("\n=== STEP 3: TRIANGLE HASH ALIGNMENT ===")
    
    alignment_result, alignment_approach = alignment_step(
        phenotype_scaled, sbs_positions, phenotype_triangles, sbs_triangles
    )

    if alignment_result.empty:
        print("Alignment failed - creating empty outputs")
        create_empty_outputs(
            snakemake.output, phenotype_scaled, scale_factor, plate, well
        )
        return

    best_alignment = alignment_result.iloc[0]
    
    # Apply transformation and save transformed positions
    phenotype_transformed = apply_transformation(
        phenotype_scaled, best_alignment
    )
    
    phenotype_transformed.to_parquet(str(snakemake.output.phenotype_transformed))
    print(f"Saved transformed positions: {snakemake.output.phenotype_transformed}")

    # Save alignment parameters
    essential_alignment = prepare_alignment_parameters(
        best_alignment, scale_factor, overlap_fraction
    )
    
    essential_alignment.to_parquet(str(snakemake.output.alignment_params))
    print(f"Saved alignment parameters: {snakemake.output.alignment_params}")

    # Step 4: Cell Merging
    print("\n=== STEP 4: CELL MERGING ===")
    
    merged_cells = cell_merging_step(
        phenotype_scaled, sbs_positions, best_alignment, threshold
    )

    if merged_cells.empty:
        print("Cell merging failed - creating empty output")
        create_empty_merged_cells_output(snakemake.output.merged_cells)
        create_failure_summary(snakemake.output.merge_summary, "no_cell_matches")
        return

    # Finalize and save results
    merged_cells_final = finalize_merged_cells(merged_cells, plate, well)
    
    merged_cells_final.to_parquet(str(snakemake.output.merged_cells))
    print(f"Saved merged cells: {snakemake.output.merged_cells}")

    # Create and save summary
    summary_stats = create_summary_statistics(
        plate, well, scale_factor, threshold, phenotype_positions,
        phenotype_scaled, sbs_positions, overlap_fraction,
        phenotype_triangles, sbs_triangles, alignment_approach,
        best_alignment, merged_cells_final
    )

    with open(str(snakemake.output.merge_summary), "w") as f:
        yaml.dump(summary_stats, f, default_flow_style=False)

    print(f"Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 ENHANCED WELL MERGE PIPELINE COMPLETED!")
    print(f"Final result: {len(merged_cells_final):,} successfully merged cells")


def coordinate_scaling_step(
    phenotype_positions: pd.DataFrame,
    scale_factor: float
) -> pd.DataFrame:
    """Perform coordinate scaling to align phenotype with SBS coordinate system.
    
    Args:
        phenotype_positions: Original phenotype cell positions
        scale_factor: Scale factor to apply to phenotype coordinates
        
    Returns:
        DataFrame with scaled phenotype coordinates
    """
    print(f"Scaling phenotype coordinates by factor: {scale_factor}")
    
    phenotype_scaled = phenotype_positions.copy()
    phenotype_scaled["i"] = phenotype_scaled["i"] * scale_factor
    phenotype_scaled["j"] = phenotype_scaled["j"] * scale_factor

    print(f"Original phenotype range: i=[{phenotype_positions['i'].min():.0f}, "
          f"{phenotype_positions['i'].max():.0f}], "
          f"j=[{phenotype_positions['j'].min():.0f}, "
          f"{phenotype_positions['j'].max():.0f}]")
    print(f"Scaled phenotype range: i=[{phenotype_scaled['i'].min():.0f}, "
          f"{phenotype_scaled['i'].max():.0f}], "
          f"j=[{phenotype_scaled['j'].min():.0f}, "
          f"{phenotype_scaled['j'].max():.0f}]")
    
    return phenotype_scaled


def validate_coordinate_overlap(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame
) -> float:
    """Validate overlap between scaled phenotype and SBS coordinates.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        
    Returns:
        Overlap fraction (0.0 to 1.0)
    """
    print(f"SBS range: i=[{sbs_positions['i'].min():.0f}, "
          f"{sbs_positions['i'].max():.0f}], "
          f"j=[{sbs_positions['j'].min():.0f}, "
          f"{sbs_positions['j'].max():.0f}]")

    # Calculate overlap
    overlap_i_min = max(phenotype_scaled["i"].min(), sbs_positions["i"].min())
    overlap_i_max = min(phenotype_scaled["i"].max(), sbs_positions["i"].max())
    overlap_j_min = max(phenotype_scaled["j"].min(), sbs_positions["j"].min())
    overlap_j_max = min(phenotype_scaled["j"].max(), sbs_positions["j"].max())

    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min

    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = ((sbs_positions["i"].max() - sbs_positions["i"].min()) * 
                   (sbs_positions["j"].max() - sbs_positions["j"].min()))
        overlap_fraction = overlap_area / sbs_area
        print(f"Coordinate overlap: {overlap_fraction:.1%} of SBS area")
    else:
        print("No coordinate overlap after scaling!")
        overlap_fraction = 0.0

    return overlap_fraction


def triangle_hashing_step(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate triangle hashes for both datasets.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        
    Returns:
        Tuple of (phenotype_triangles, sbs_triangles) DataFrames
    """
    print("Generating triangle hash for scaled phenotype...")
    phenotype_triangles = well_level_triangle_hash(phenotype_scaled)

    print("Generating triangle hash for SBS...")
    sbs_triangles = well_level_triangle_hash(sbs_positions)

    print(f"Generated {len(phenotype_triangles)} scaled phenotype triangles")
    print(f"Generated {len(sbs_triangles)} SBS triangles")
    
    return phenotype_triangles, sbs_triangles


def alignment_step(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    phenotype_triangles: pd.DataFrame,
    sbs_triangles: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Perform alignment using triangle hash or identity fallback.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        phenotype_triangles: Phenotype triangle hashes
        sbs_triangles: SBS triangle hashes
        
    Returns:
        Tuple of (alignment_result, approach_name)
    """
    alignment_approach = "unknown"
    
    # Try triangle hash alignment first
    print("Attempting triangle hash alignment...")
    try:
        alignment_result = triangle_hash_well_alignment(
            phenotype_positions=phenotype_scaled,
            sbs_positions=sbs_positions,
            max_cells_for_hash=75000,
            threshold_triangle=0.1,
            threshold_point=2.0,
            min_score=0.05,
        )

        if not alignment_result.empty:
            best_alignment = alignment_result.iloc[0]
            print(f"Triangle hash alignment successful:")
            print(f"   Score: {best_alignment.get('score', 0):.3f}")
            print(f"   Determinant: {best_alignment.get('determinant', 0):.6f}")
            alignment_approach = "triangle_hash_after_scaling"
            return alignment_result, alignment_approach

    except Exception as e:
        print(f"Triangle hash alignment failed: {e}")

    # Fallback to identity transformation
    print("Falling back to identity transformation (no additional changes)...")
    print("Assuming coordinates are already properly aligned after scaling.")
    
    alignment_result, alignment_approach = create_identity_alignment(
        phenotype_scaled, sbs_positions
    )
    
    return alignment_result, alignment_approach


def create_identity_alignment(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    sample_size: int = 10000
) -> tuple[pd.DataFrame, str]:
    """Create identity transformation alignment as fallback.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        sample_size: Number of cells to sample for validation
        
    Returns:
        Tuple of (alignment_result, approach_name)
    """
    from scipy.spatial.distance import cdist
    
    # Sample cells for validation
    if len(phenotype_scaled) > sample_size:
        pheno_sample = phenotype_scaled.sample(n=sample_size)[["i", "j"]].values
    else:
        pheno_sample = phenotype_scaled[["i", "j"]].values

    if len(sbs_positions) > sample_size:
        sbs_sample = sbs_positions.sample(n=sample_size)[["i", "j"]].values
    else:
        sbs_sample = sbs_positions[["i", "j"]].values

    # Validate identity transformation (no translation/rotation)
    distances = cdist(pheno_sample, sbs_sample, metric="euclidean")
    min_distances = distances.min(axis=1)
    score = (min_distances < 10.0).mean()

    print(f"Identity transformation validation score: {score:.3f}")
    print(f"Mean nearest neighbor distance: {min_distances.mean():.2f} px")

    # Calculate overlap for identity case
    overlap_i_min = max(phenotype_scaled["i"].min(), sbs_positions["i"].min())
    overlap_i_max = min(phenotype_scaled["i"].max(), sbs_positions["i"].max())
    overlap_j_min = max(phenotype_scaled["j"].min(), sbs_positions["j"].min()) 
    overlap_j_max = min(phenotype_scaled["j"].max(), sbs_positions["j"].max())

    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min
    
    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = ((sbs_positions["i"].max() - sbs_positions["i"].min()) * 
                   (sbs_positions["j"].max() - sbs_positions["j"].min()))
        overlap_fraction = overlap_area / sbs_area if sbs_area > 0 else 0
    else:
        overlap_fraction = 0.0

    best_alignment = {
        "rotation": np.eye(2),
        "translation": np.array([0.0, 0.0]),
        "score": score,
        "determinant": 1.0,
        "transformation_type": "identity_after_scaling",
        "approach": "identity_fallback",
        "overlap_fraction": overlap_fraction,
        "validation_mean_distance": min_distances.mean(),
        "validation_median_distance": np.median(min_distances),
        "has_overlap": has_overlap,
    }

    alignment_result = pd.DataFrame([best_alignment])
    approach = "identity_fallback"
    
    return alignment_result, approach


def apply_transformation(
    phenotype_scaled: pd.DataFrame,
    alignment: pd.Series
) -> pd.DataFrame:
    """Apply transformation to phenotype coordinates.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        alignment: Alignment parameters
        
    Returns:
        DataFrame with transformed phenotype positions
    """
    # Extract transformation parameters with safe conversions
    rotation_matrix = alignment.get("rotation", np.eye(2))
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.eye(2)
    else:
        rotation_matrix = np.array(rotation_matrix).reshape(2, 2)

    translation_vector = alignment.get("translation", np.array([0.0, 0.0]))
    if not isinstance(translation_vector, np.ndarray):
        translation_vector = np.array([0.0, 0.0])
    else:
        translation_vector = np.array(translation_vector).flatten()[:2]

    # Apply transformation
    pheno_coords_scaled = phenotype_scaled[["i", "j"]].values
    pheno_coords_transformed = (
        pheno_coords_scaled @ rotation_matrix.T + translation_vector
    )

    phenotype_transformed = phenotype_scaled.copy()
    phenotype_transformed["i_transformed"] = pheno_coords_transformed[:, 0]
    phenotype_transformed["j_transformed"] = pheno_coords_transformed[:, 1]

    return phenotype_transformed


def cell_merging_step(
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    alignment: pd.Series,
    threshold: float
) -> pd.DataFrame:
    """Perform cell-level merging using alignment parameters.
    
    Args:
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        alignment: Alignment parameters
        threshold: Distance threshold for matching
        
    Returns:
        DataFrame with merged cell pairs
    """
    print(f"Using distance threshold: {threshold} pixels")

    merged_cells = merge_stitched_cells(
        phenotype_positions=phenotype_scaled,
        sbs_positions=sbs_positions,
        alignment=alignment,
        threshold=threshold,
        chunk_size=50000,
        output_path=str(snakemake.output.merged_cells),
    )

    if not merged_cells.empty:
        print(f"Cell merging successful:")
        print(f"   Total matches: {len(merged_cells):,}")
        print(f"   Mean distance: {merged_cells['distance'].mean():.2f} px")
        print(f"   Max distance: {merged_cells['distance'].max():.2f} px")
        
        distance_stats = [
            (5, (merged_cells['distance'] < 5).sum()),
            (10, (merged_cells['distance'] < 10).sum())
        ]
        
        for thresh, count in distance_stats:
            pct = (count / len(merged_cells)) * 100
            print(f"   Matches < {thresh}px: {count:,} ({pct:.1f}%)")

    return merged_cells


def finalize_merged_cells(
    merged_cells: pd.DataFrame,
    plate: str,
    well: str
) -> pd.DataFrame:
    """Add metadata and reorder columns for final output.
    
    Args:
        merged_cells: Raw merged cell data
        plate: Plate identifier
        well: Well identifier
        
    Returns:
        DataFrame with finalized merged cells
    """
    # Add plate and well columns
    merged_cells["plate"] = plate
    merged_cells["well"] = well

    # Define expected column order
    output_columns = [
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance"
    ]

    # Only include columns that exist
    available_columns = [col for col in output_columns if col in merged_cells.columns]
    merged_cells_final = merged_cells[available_columns]

    print(f"Final merged cells:")
    print(f"   Columns: {list(merged_cells_final.columns)}")
    print(f"   Rows: {len(merged_cells_final):,}")

    return merged_cells_final


def prepare_alignment_parameters(
    alignment: pd.Series,
    scale_factor: float,
    overlap_fraction: float
) -> pd.DataFrame:
    """Prepare alignment parameters for Parquet serialization.
    
    Args:
        alignment: Raw alignment parameters
        scale_factor: Scale factor used
        overlap_fraction: Coordinate overlap fraction
        
    Returns:
        DataFrame with serializable alignment parameters
    """
    def safe_float(value: any, default: float = 0.0) -> float:
        """Safely convert value to float with fallback."""
        try:
            if isinstance(value, str) and value.lower() in ["n/a", "na", "none"]:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    # Extract transformation matrices safely
    rotation_matrix = alignment.get("rotation", np.eye(2))
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.eye(2)
    else:
        rotation_matrix = np.array(rotation_matrix).reshape(2, 2)

    translation_vector = alignment.get("translation", np.array([0.0, 0.0]))
    if not isinstance(translation_vector, np.ndarray):
        translation_vector = np.array([0.0, 0.0])
    else:
        translation_vector = np.array(translation_vector).flatten()[:2]

    # Get scores by threshold safely
    scores_by_threshold = alignment.get("scores_by_threshold", {})
    if not isinstance(scores_by_threshold, dict):
        scores_by_threshold = {}

    essential_alignment = pd.DataFrame([{
        "rotation_matrix_flat": rotation_matrix.flatten().tolist(),
        "translation_vector": translation_vector.tolist(),
        "scale_factor": safe_float(alignment.get("scale_factor", scale_factor)),
        "score": safe_float(alignment.get("score", 0)),
        "determinant": safe_float(alignment.get("determinant", 1)),
        "transformation_type": str(alignment.get("transformation_type", "unknown")),
        "approach": str(alignment.get("approach", "unknown")),
        "overlap_fraction": safe_float(alignment.get("overlap_fraction", overlap_fraction)),
        "validation_mean_distance": safe_float(alignment.get("validation_mean_distance", 0.0)),
        "validation_median_distance": safe_float(alignment.get("validation_median_distance", 0.0)),
        "has_overlap": bool(alignment.get("has_overlap", True)),
        # Individual threshold scores
        "score_2px": safe_float(scores_by_threshold.get(2, 0.0)),
        "score_5px": safe_float(scores_by_threshold.get(5, 0.0)),
        "score_10px": safe_float(scores_by_threshold.get(10, 0.0)),
        "score_20px": safe_float(scores_by_threshold.get(20, 0.0)),
        "score_50px": safe_float(scores_by_threshold.get(50, 0.0)),
    }])

    return essential_alignment


def create_summary_statistics(
    plate: str,
    well: str,
    scale_factor: float,
    threshold: float,
    phenotype_positions: pd.DataFrame,
    phenotype_scaled: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    overlap_fraction: float,
    phenotype_triangles: pd.DataFrame,
    sbs_triangles: pd.DataFrame,
    alignment_approach: str,
    best_alignment: pd.Series,
    merged_cells_final: pd.DataFrame
) -> dict:
    """Create comprehensive summary statistics for the merge process.
    
    Args:
        plate: Plate identifier
        well: Well identifier
        scale_factor: Scale factor used
        threshold: Distance threshold used
        phenotype_positions: Original phenotype positions
        phenotype_scaled: Scaled phenotype positions
        sbs_positions: SBS positions
        overlap_fraction: Coordinate overlap fraction
        phenotype_triangles: Generated phenotype triangles
        sbs_triangles: Generated SBS triangles
        alignment_approach: Alignment approach used
        best_alignment: Best alignment parameters
        merged_cells_final: Final merged cells
        
    Returns:
        Dictionary containing comprehensive summary statistics
    """
    def safe_float_conversion(value: any) -> float:
        """Safely convert values to float for YAML serialization."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    summary_stats = {
        "status": "success",
        "plate": plate,
        "well": well,
        "processing_parameters": {
            "scale_factor": float(scale_factor),
            "distance_threshold_pixels": float(threshold),
        },
        "input_data": {
            "phenotype_cells_total": len(phenotype_positions),
            "phenotype_cells_scaled": len(phenotype_scaled),
            "sbs_cells_total": len(sbs_positions),
            "coordinate_overlap_fraction": float(overlap_fraction),
        },
        "step1_coordinate_scaling": {
            "scale_factor": float(scale_factor),
            "phenotype_range_original": {
                "i_min": safe_float_conversion(phenotype_positions["i"].min()),
                "i_max": safe_float_conversion(phenotype_positions["i"].max()),
                "j_min": safe_float_conversion(phenotype_positions["j"].min()),
                "j_max": safe_float_conversion(phenotype_positions["j"].max()),
            },
            "phenotype_range_scaled": {
                "i_min": safe_float_conversion(phenotype_scaled["i"].min()),
                "i_max": safe_float_conversion(phenotype_scaled["i"].max()),
                "j_min": safe_float_conversion(phenotype_scaled["j"].min()),
                "j_max": safe_float_conversion(phenotype_scaled["j"].max()),
            },
            "overlap_fraction": float(overlap_fraction),
        },
        "step2_triangle_hashing": {
            "phenotype_triangles": len(phenotype_triangles),
            "sbs_triangles": len(sbs_triangles),
            "coordinates_used": "scaled_phenotype_and_original_sbs",
        },
        "step3_alignment": {
            "approach_used": alignment_approach,
            "scale_factor_used": safe_float_conversion(best_alignment.get("scale_factor", scale_factor)),
            "alignment_score": safe_float_conversion(best_alignment.get("score", 0)),
            "determinant": safe_float_conversion(best_alignment.get("determinant", 1)),
            "transformation_type": str(best_alignment.get("transformation_type", "unknown")),
        },
        "step4_cell_merging": {
            "merged_cells_count": len(merged_cells_final),
            "mean_match_distance": safe_float_conversion(merged_cells_final["distance"].mean()),
            "max_match_distance": safe_float_conversion(merged_cells_final["distance"].max()),
            "matches_under_5px": int((merged_cells_final["distance"] < 5).sum()),
            "matches_under_10px": int((merged_cells_final["distance"] < 10).sum()),
            "match_rate_phenotype": safe_float_conversion(len(merged_cells_final) / len(phenotype_positions)),
            "match_rate_sbs": safe_float_conversion(len(merged_cells_final) / len(sbs_positions)),
        },
    }

    return summary_stats


def create_empty_outputs(
    outputs: object,
    phenotype_scaled: pd.DataFrame,
    scale_factor: float,
    plate: str,
    well: str
) -> None:
    """Create empty output files when pipeline fails early.
    
    Args:
        outputs: Snakemake output object
        phenotype_scaled: Scaled phenotype positions (may be partial)
        scale_factor: Scale factor used
        plate: Plate identifier
        well: Well identifier
    """
    # Create empty merged cells
    create_empty_merged_cells_output(outputs.merged_cells)

    # Create empty triangle files
    empty_triangles = pd.DataFrame(columns=["V_0", "V_1", "c_0", "c_1", "magnitude"])
    empty_triangles.to_parquet(str(outputs.phenotype_triangles))
    empty_triangles.to_parquet(str(outputs.sbs_triangles))

    # Save scaled positions (even on failure)
    phenotype_scaled.to_parquet(str(outputs.phenotype_scaled))

    # Save identity-transformed positions
    phenotype_transformed = phenotype_scaled.copy()
    phenotype_transformed["i_transformed"] = phenotype_scaled["i"]
    phenotype_transformed["j_transformed"] = phenotype_scaled["j"]
    phenotype_transformed.to_parquet(str(outputs.phenotype_transformed))

    # Create empty alignment parameters
    empty_alignment = pd.DataFrame([{
        "rotation_matrix_flat": [1.0, 0.0, 0.0, 1.0],
        "translation_vector": [0.0, 0.0],
        "score": 0.0,
        "determinant": 1.0,
        "transformation_type": "failed_pipeline_step",
        "scale_factor": float(scale_factor),
        "approach": "pipeline_failed",
        "overlap_fraction": 0.0,
        "validation_mean_distance": 0.0,
        "validation_median_distance": 0.0,
        "has_overlap": False,
        "score_2px": 0.0,
        "score_5px": 0.0,
        "score_10px": 0.0,
        "score_20px": 0.0,
        "score_50px": 0.0,
    }])
    empty_alignment.to_parquet(str(outputs.alignment_params))

    # Create failure summary
    create_failure_summary(outputs.merge_summary, "insufficient_triangles")


def create_empty_merged_cells_output(output_path: str) -> None:
    """Create empty merged cells output file.
    
    Args:
        output_path: Path to save empty merged cells file
    """
    empty_df = pd.DataFrame(columns=[
        "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance"
    ])
    empty_df.to_parquet(str(output_path))


def create_failure_summary(output_path: str, reason: str) -> None:
    """Create failure summary file.
    
    Args:
        output_path: Path to save summary file
        reason: Reason for failure
    """
    with open(str(output_path), "w") as f:
        yaml.dump({"status": "failed", "reason": reason}, f)


if __name__ == "__main__":
    main()