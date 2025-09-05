"""Well Alignment - Coordinate scaling, triangle hashing, and alignment estimation.

This script performs the first step of the well-level merge pipeline:
1. Auto-calculates scale factor based on coordinate ranges
2. Scales phenotype coordinates to match SBS coordinate system
3. Generates triangle hash features for both datasets
4. Performs adaptive regional triangle hash alignment
5. Saves alignment parameters and transformed coordinates
6. Outputs alignment summary in TSV format for multi-well analysis

The alignment process uses triangle-based feature matching with RANSAC to
estimate robust transformation parameters between imaging modalities.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.spatial.distance import cdist

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_alignment import (
    calculate_scale_factor_from_positions,
    scale_coordinates,
    well_level_triangle_hash,
    triangle_hash_well_alignment,
    calculate_coordinate_overlap,
)

print("=== WELL ALIGNMENT ===")

# Load cell positions - these now come from the stitching outputs at index [0]
# Input references: phenotype_positions and sbs_positions (named inputs from rules)
phenotype_positions = validate_dtypes(
    pd.read_parquet(snakemake.input.phenotype_positions)
)
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

plate = snakemake.params.plate
well = snakemake.params.well
score = snakemake.params.score

print(f"Processing Plate {plate}, Well {well}")
print(f"Phenotype cells: {len(phenotype_positions):,}")
print(f"SBS cells: {len(sbs_positions):,}")

# =================================================================
# AUTO-CALCULATE SCALE FACTOR AND COORDINATE SCALING
# =================================================================
print("\n--- Auto-calculating Scale Factor from Position Ranges ---")

# Calculate scale factor from coordinate ranges
scale_factor = calculate_scale_factor_from_positions(phenotype_positions, sbs_positions)

print(f"Auto-calculated scale factor: {scale_factor:.6f}")

phenotype_scaled = scale_coordinates(phenotype_positions, scale_factor=scale_factor)

print(
    f"Original phenotype range: i=[{phenotype_positions['i'].min():.0f}, {phenotype_positions['i'].max():.0f}], j=[{phenotype_positions['j'].min():.0f}, {phenotype_positions['j'].max():.0f}]"
)
print(
    f"Scaled phenotype range: i=[{phenotype_scaled['i'].min():.0f}, {phenotype_scaled['i'].max():.0f}], j=[{phenotype_scaled['j'].min():.0f}, {phenotype_scaled['j'].max():.0f}]"
)
print(
    f"SBS range: i=[{sbs_positions['i'].min():.0f}, {sbs_positions['i'].max():.0f}], j=[{sbs_positions['j'].min():.0f}, {sbs_positions['j'].max():.0f}]"
)

# Calculate overlap
overlap_fraction = calculate_coordinate_overlap(phenotype_scaled, sbs_positions)
print(f"Coordinate overlap: {overlap_fraction:.1%} of SBS area")

# Save scaled coordinates - output index [0]
phenotype_scaled.to_parquet(str(snakemake.output.scaled_phenotype_positions))
print(
    f"✅ Saved scaled phenotype positions: {snakemake.output.scaled_phenotype_positions}"
)

# =================================================================
# TRIANGLE HASHING
# =================================================================
print("\n--- Triangle Hashing ---")

# Generate triangle hashes using scaled coordinates
print("Generating triangle hash for scaled phenotype...")
phenotype_triangles = well_level_triangle_hash(phenotype_scaled)

print("Generating triangle hash for SBS...")
sbs_triangles = well_level_triangle_hash(sbs_positions)

if len(phenotype_triangles) == 0 or len(sbs_triangles) == 0:
    print("❌ Triangle hash generation failed")

    # Save empty triangle files - outputs index [1] and [2]
    empty_triangles = pd.DataFrame(columns=["V_0", "V_1", "c_0", "c_1", "magnitude"])
    empty_triangles.to_parquet(str(snakemake.output.phenotype_triangles))
    empty_triangles.to_parquet(str(snakemake.output.sbs_triangles))

    # Create failed alignment with identity transformation - output index [3]
    failed_alignment_data = {
        "rotation_matrix_flat": [1.0, 0.0, 0.0, 1.0],
        "translation_vector": [0.0, 0.0],
        "score": 0.0,
        "determinant": 1.0,
        "transformation_type": "failed_insufficient_triangles",
        "scale_factor": float(scale_factor),
        "approach": "failed",
        "overlap_fraction": 0.0,
        "validation_mean_distance": 0.0,
        "validation_median_distance": 0.0,
        "has_overlap": False,
    }

    failed_alignment = pd.DataFrame([failed_alignment_data])
    failed_alignment.to_parquet(str(snakemake.output.alignment_params))

    # Create failure summary in TSV format - output index [4]
    def safe_float(value, default: float = 0.0, precision: int = 6) -> float:
        try:
            if value is None:
                return default
            return round(float(value), precision)
        except (ValueError, TypeError):
            return default

    def safe_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    summary = {
        # Well identifiers
        "plate": str(plate),
        "well": str(well),
        # Status and basic metrics
        "status": "failed",
        "failure_reason": "insufficient_triangles",
        "scale_factor": safe_float(scale_factor),
        "overlap_fraction": safe_float(overlap_fraction, precision=3),
        # Triangle generation
        "phenotype_triangles": 0,
        "sbs_triangles": 0,
        # Alignment parameters
        "threshold_triangle": 0.3,
        "score_threshold": safe_float(score, precision=3),
        "threshold_point": 2.0,
        # Alignment results
        "approach": "failed",
        "transformation_type": "failed_insufficient_triangles",
        "alignment_score": 0.0,
        "determinant": 1.0,
        # Transformation matrix components
        "rotation_r00": 1.0,
        "rotation_r01": 0.0,
        "rotation_r10": 0.0,
        "rotation_r11": 1.0,
        "translation_tx": 0.0,
        "translation_ty": 0.0,
        # Validation metrics
        "validation_mean_distance": 0.0,
        "validation_median_distance": 0.0,
        # Regional sampling details
        "region_size": 0.0,
        "sampling_attempts": 0,
        "triangles_matched": 0,
    }

    # Save failure summary as TSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        str(snakemake.output.alignment_summary),
        sep="\t",
        index=False,
        float_format="%.6g",
    )

    # Create empty transformed positions - output index [5]
    empty_transformed = phenotype_positions.copy()
    empty_transformed.to_parquet(str(snakemake.output.transformed_phenotype_positions))

else:
    print(f"✅ Generated {len(phenotype_triangles)} phenotype triangles")
    print(f"✅ Generated {len(sbs_triangles)} SBS triangles")

    # Save triangle hashes - outputs index [1] and [2]
    phenotype_triangles.to_parquet(str(snakemake.output.phenotype_triangles))
    sbs_triangles.to_parquet(str(snakemake.output.sbs_triangles))

    # =================================================================
    # ADAPTIVE REGIONAL TRIANGLE HASHING
    # =================================================================
    print("\n--- Adaptive Regional Triangle Hashing ---")

    try:
        alignment_result = triangle_hash_well_alignment(
            phenotype_positions=phenotype_scaled,
            sbs_positions=sbs_positions,
            max_cells_for_hash=75000,
            threshold_triangle=0.3,
            threshold_point=2.0,
            score=score,
            adaptive_region=True,
            initial_region_size=7000,
            min_triangles=100,
        )

        if not alignment_result.empty:
            best_alignment = alignment_result.iloc[0]

            # Check alignment quality (only score-based validation)
            alignment_score = best_alignment.get("score", 0)
            score_ok = alignment_score >= score

            if score_ok:
                print(f"✅ Regional triangle hash alignment successful:")
                print(f"   Score: {alignment_score:.3f} (min required: {score})")
                print(f"   Determinant: {best_alignment.get('determinant', 'N/A'):.6f}")
                print(
                    f"   Region size: {best_alignment.get('final_region_size', 'unknown')}"
                )
                print(f"   Attempts: {best_alignment.get('attempts', 'unknown')}")
                alignment_status = "success"
            else:
                print(f"⚠️  Alignment quality issues:")
                print(f"   Score: {alignment_score:.3f} (min required: {score})")
                alignment_status = "low_score"
        else:
            print("❌ Regional triangle hash alignment returned empty result")
            alignment_result = None
            alignment_status = "failed"

    except Exception as e:
        print(f"❌ Regional triangle hash alignment failed: {e}")
        alignment_result = None
        alignment_status = "failed"

    # Fallback to identity transformation
    if alignment_result is None or alignment_result.empty:
        print("Using identity transformation fallback...")

        # Validate with a sample to estimate alignment quality
        sample_size = min(1000, len(phenotype_scaled), len(sbs_positions))
        pheno_sample = phenotype_scaled.sample(n=sample_size)[["i", "j"]].values
        sbs_sample = sbs_positions.sample(n=sample_size)[["i", "j"]].values

        distances = cdist(pheno_sample, sbs_sample, metric="euclidean")
        min_distances = distances.min(axis=1)
        score_val = (min_distances < 10.0).mean()

        best_alignment = {
            "rotation": np.eye(2),
            "translation": np.array([0.0, 0.0]),
            "score": score_val,
            "determinant": 1.0,
            "transformation_type": "identity_after_scaling",
            "scale_factor": scale_factor,
            "approach": "identity_fallback",
            "validation_mean_distance": min_distances.mean(),
            "validation_median_distance": np.median(min_distances),
            "has_overlap": True,
        }
        alignment_status = "identity_fallback"
        alignment_result = pd.DataFrame([best_alignment])
    else:
        best_alignment = alignment_result.iloc[0]

    # Prepare alignment parameters for saving
    rotation_matrix = best_alignment.get("rotation", np.eye(2))
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.eye(2)

    translation_vector = best_alignment.get("translation", np.array([0.0, 0.0]))
    if not isinstance(translation_vector, np.ndarray):
        translation_vector = np.array([0.0, 0.0])

    def safe_float(value, default: float = 0.0) -> float:
        """Safely convert value to float with fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    # Convert to standard Python lists for Parquet compatibility
    rotation_flat = rotation_matrix.flatten().astype(float).tolist()
    translation_list = translation_vector.astype(float).tolist()

    # Debug output
    print(f"Saving alignment parameters:")
    print(f"  Rotation matrix: {rotation_matrix}")
    print(f"  Rotation flat: {rotation_flat}")
    print(f"  Translation: {translation_vector}")
    print(f"  Translation list: {translation_list}")

    essential_alignment_data = {
        "rotation_matrix_flat": rotation_flat,
        "translation_vector": translation_list,
        "scale_factor": safe_float(best_alignment.get("scale_factor", scale_factor)),
        "score": safe_float(best_alignment.get("score", 0)),
        "determinant": safe_float(best_alignment.get("determinant", 1)),
        "transformation_type": str(
            best_alignment.get("transformation_type", "unknown")
        ),
        "approach": str(best_alignment.get("approach", "unknown")),
        "validation_mean_distance": safe_float(
            best_alignment.get("validation_mean_distance", 0.0)
        ),
        "validation_median_distance": safe_float(
            best_alignment.get("validation_median_distance", 0.0)
        ),
        "has_overlap": bool(best_alignment.get("has_overlap", True)),
    }

    essential_alignment = pd.DataFrame([essential_alignment_data])
    # Save alignment parameters - output index [3]
    essential_alignment.to_parquet(str(snakemake.output.alignment_params))

    print(f"✅ Saved alignment parameters: {snakemake.output.alignment_params}")

    # Create alignment summary in TSV format
    def safe_float_summary(value, default: float = 0.0, precision: int = 6) -> float:
        """Safely convert value to float with specified precision."""
        try:
            if value is None:
                return default
            return round(float(value), precision)
        except (ValueError, TypeError):
            return default

    def safe_int_summary(value, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    # Extract rotation matrix components
    if isinstance(rotation_matrix, np.ndarray):
        r00, r01 = float(rotation_matrix[0, 0]), float(rotation_matrix[0, 1])
        r10, r11 = float(rotation_matrix[1, 0]), float(rotation_matrix[1, 1])
    else:
        r00, r01, r10, r11 = 1.0, 0.0, 0.0, 1.0

    # Extract translation vector components
    if isinstance(translation_vector, np.ndarray):
        tx, ty = float(translation_vector[0]), float(translation_vector[1])
    else:
        tx, ty = 0.0, 0.0

    summary = {
        # Well identifiers
        "plate": str(plate),
        "well": str(well),
        # Status and basic metrics
        "status": str(alignment_status),
        "failure_reason": "",
        "scale_factor": safe_float_summary(scale_factor),
        "overlap_fraction": safe_float_summary(overlap_fraction, precision=3),
        # Triangle generation
        "phenotype_triangles": safe_int_summary(len(phenotype_triangles)),
        "sbs_triangles": safe_int_summary(len(sbs_triangles)),
        # Alignment parameters
        "threshold_triangle": 0.3,  # Fixed parameter
        "score_threshold": safe_float_summary(score, precision=3),
        "threshold_point": 2.0,  # Fixed parameter
        # Alignment results
        "approach": str(best_alignment.get("approach", "")),
        "transformation_type": str(best_alignment.get("transformation_type", "")),
        "alignment_score": safe_float_summary(
            best_alignment.get("score", 0), precision=3
        ),
        "determinant": safe_float_summary(
            best_alignment.get("determinant", 1), precision=6
        ),
        # Transformation matrix components
        "rotation_r00": safe_float_summary(r00, precision=6),
        "rotation_r01": safe_float_summary(r01, precision=6),
        "rotation_r10": safe_float_summary(r10, precision=6),
        "rotation_r11": safe_float_summary(r11, precision=6),
        "translation_tx": safe_float_summary(tx, precision=3),
        "translation_ty": safe_float_summary(ty, precision=3),
        # Validation metrics
        "validation_mean_distance": safe_float_summary(
            best_alignment.get("validation_mean_distance", 0), precision=3
        ),
        "validation_median_distance": safe_float_summary(
            best_alignment.get("validation_median_distance", 0), precision=3
        ),
        # Regional sampling details
        "region_size": safe_float_summary(
            best_alignment.get("final_region_size", 0), precision=0
        ),
        "sampling_attempts": safe_int_summary(best_alignment.get("attempts", 0)),
        "triangles_matched": safe_int_summary(
            best_alignment.get("triangles_matched", 0)
        ),
    }

    # Save alignment summary as TSV - output index [4]
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        str(snakemake.output.alignment_summary),
        sep="\t",
        index=False,
        float_format="%.6g",
    )
    print(f"✅ Saved alignment summary: {snakemake.output.alignment_summary}")

    # =================================================================
    # GENERATE TRANSFORMED COORDINATES
    # =================================================================
    print("\n--- Generating Transformed Coordinates ---")

    # Apply the transformation to the scaled phenotype coordinates
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.eye(2)
    if not isinstance(translation_vector, np.ndarray):
        translation_vector = np.array([0.0, 0.0])

    print(f"Applying transformation:")
    print(f"  Rotation: {rotation_matrix}")
    print(f"  Translation: {translation_vector}")

    # Transform the coordinates
    pheno_coords = phenotype_scaled[["i", "j"]].values
    transformed_coords = pheno_coords @ rotation_matrix.T + translation_vector

    # Create transformed dataframe
    phenotype_transformed = phenotype_scaled.copy()
    phenotype_transformed["i"] = transformed_coords[:, 0]
    phenotype_transformed["j"] = transformed_coords[:, 1]

    print(f"Coordinate transformation applied:")
    print(
        f"  Original range: i=[{phenotype_scaled['i'].min():.0f}, {phenotype_scaled['i'].max():.0f}], j=[{phenotype_scaled['j'].min():.0f}, {phenotype_scaled['j'].max():.0f}]"
    )
    print(
        f"  Transformed range: i=[{phenotype_transformed['i'].min():.0f}, {phenotype_transformed['i'].max():.0f}], j=[{phenotype_transformed['j'].min():.0f}, {phenotype_transformed['j'].max():.0f}]"
    )

    # Save transformed coordinates - output index [5]
    phenotype_transformed.to_parquet(
        str(snakemake.output.transformed_phenotype_positions)
    )
    print(
        f"✅ Saved transformed phenotype positions: {snakemake.output.transformed_phenotype_positions}"
    )

print(f"\n🎉 Alignment completed successfully!")