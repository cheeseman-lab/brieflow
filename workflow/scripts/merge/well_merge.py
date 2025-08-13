"""
Enhanced well merge pipeline with separate output files for each step.
Save this as: workflow/scripts/merge/well_merge.py

This version saves triangle hashes, alignment parameters, and summary as separate 
Snakemake outputs instead of creating subdirectories.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import (
    well_level_triangle_hash,
    pure_scaling_alignment,
    merge_stitched_cells,
    triangle_hash_well_alignment
)

def main():
    print("=== ENHANCED WELL MERGE PIPELINE (WITH SEPARATE OUTPUTS) ===")
    
    # Load stitched cell positions
    phenotype_positions = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_positions))
    sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))
    
    plate = snakemake.params.plate
    well = snakemake.params.well
    scale_factor = snakemake.params.scale_factor
    threshold = snakemake.params.threshold
    
    print(f"Processing Plate {plate}, Well {well}")
    print(f"Phenotype cells: {len(phenotype_positions):,}")
    print(f"SBS cells: {len(sbs_positions):,}")
    print(f"Scale factor: {scale_factor}")
    print(f"Distance threshold: {threshold} px")
    
    # =================================================================
    # STEP 1: COORDINATE SCALING (DO THIS FIRST!)
    # =================================================================
    print("\n=== STEP 1: COORDINATE SCALING ===")
    
    # Scale phenotype coordinates to SBS coordinate system BEFORE triangle hashing
    print(f"Scaling phenotype coordinates by factor: {scale_factor}")
    phenotype_scaled = phenotype_positions.copy()
    phenotype_scaled['i'] = phenotype_scaled['i'] * scale_factor
    phenotype_scaled['j'] = phenotype_scaled['j'] * scale_factor
    
    print(f"Original phenotype range: i=[{phenotype_positions['i'].min():.0f}, {phenotype_positions['i'].max():.0f}], j=[{phenotype_positions['j'].min():.0f}, {phenotype_positions['j'].max():.0f}]")
    print(f"Scaled phenotype range: i=[{phenotype_scaled['i'].min():.0f}, {phenotype_scaled['i'].max():.0f}], j=[{phenotype_scaled['j'].min():.0f}, {phenotype_scaled['j'].max():.0f}]")
    print(f"SBS range: i=[{sbs_positions['i'].min():.0f}, {sbs_positions['i'].max():.0f}], j=[{sbs_positions['j'].min():.0f}, {sbs_positions['j'].max():.0f}]")
    
    # Calculate overlap
    overlap_i_min = max(phenotype_scaled['i'].min(), sbs_positions['i'].min())
    overlap_i_max = min(phenotype_scaled['i'].max(), sbs_positions['i'].max())
    overlap_j_min = max(phenotype_scaled['j'].min(), sbs_positions['j'].min())
    overlap_j_max = min(phenotype_scaled['j'].max(), sbs_positions['j'].max())
    
    has_overlap = overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min
    
    if has_overlap:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_area = (sbs_positions['i'].max() - sbs_positions['i'].min()) * (sbs_positions['j'].max() - sbs_positions['j'].min())
        overlap_fraction = overlap_area / sbs_area
        print(f"✅ Coordinate overlap: {overlap_fraction:.1%} of SBS area")
    else:
        print("❌ No coordinate overlap after scaling!")
        overlap_fraction = 0.0
    
    # SAVE SCALED PHENOTYPE POSITIONS
    phenotype_scaled.to_parquet(str(snakemake.output.phenotype_scaled))
    print(f"✅ Saved scaled phenotype positions: {snakemake.output.phenotype_scaled}")
    
    # =================================================================
    # STEP 2: TRIANGLE HASHING (IN SCALED COORDINATE SYSTEM)
    # =================================================================
    print("\n=== STEP 2: TRIANGLE HASHING (SCALED COORDINATES) ===")
    
    # Generate triangle hashes using SCALED phenotype coordinates
    print("Generating triangle hash for scaled phenotype...")
    phenotype_triangles = well_level_triangle_hash(phenotype_scaled)  # Use scaled coordinates!
    
    print("Generating triangle hash for SBS...")
    sbs_triangles = well_level_triangle_hash(sbs_positions)
    
    if len(phenotype_triangles) == 0 or len(sbs_triangles) == 0:
        print("❌ Triangle hash generation failed - insufficient triangles")
        # Create empty outputs
        empty_df = pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
        empty_df.to_parquet(str(snakemake.output.merged_cells))
        
        # Create empty triangle files
        pd.DataFrame(columns=['V_0', 'V_1', 'c_0', 'c_1', 'magnitude']).to_parquet(str(snakemake.output.phenotype_triangles))
        pd.DataFrame(columns=['V_0', 'V_1', 'c_0', 'c_1', 'magnitude']).to_parquet(str(snakemake.output.sbs_triangles))
        
        # ADD THIS: Save scaled positions even on failure
        phenotype_scaled.to_parquet(str(snakemake.output.phenotype_scaled))
        
        # ADD THIS: Save identity-transformed positions (no actual transformation)
        phenotype_transformed = phenotype_scaled.copy()
        phenotype_transformed['i_transformed'] = phenotype_scaled['i']
        phenotype_transformed['j_transformed'] = phenotype_scaled['j']
        phenotype_transformed.to_parquet(str(snakemake.output.phenotype_transformed))
        
        # Create empty alignment file
        empty_alignment = pd.DataFrame([{
            'rotation_matrix_flat': [1.0, 0.0, 0.0, 1.0],
            'translation_vector': [0.0, 0.0],
            'score': 0.0,
            'determinant': 1.0,
            'transformation_type': 'failed_triangulation',
            'scale_factor': float(scale_factor),
            'approach': 'triangulation_failed',
            'overlap_fraction': 0.0,
            'validation_mean_distance': 0.0,
            'validation_median_distance': 0.0,
            'has_overlap': False,
            'score_2px': 0.0,
            'score_5px': 0.0,
            'score_10px': 0.0,
            'score_20px': 0.0,
            'score_50px': 0.0
        }])
        empty_alignment.to_parquet(str(snakemake.output.alignment_params))
        
        # Create empty summary
        with open(str(snakemake.output.merge_summary), 'w') as f:
            yaml.dump({'status': 'failed', 'reason': 'insufficient_triangles'}, f)
        return
    
    print(f"✅ Generated {len(phenotype_triangles)} scaled phenotype triangles")
    print(f"✅ Generated {len(sbs_triangles)} SBS triangles")
    
    # Save triangle hashes as separate outputs
    phenotype_triangles.to_parquet(str(snakemake.output.phenotype_triangles))
    sbs_triangles.to_parquet(str(snakemake.output.sbs_triangles))
    
    print(f"✅ Saved scaled phenotype triangles: {snakemake.output.phenotype_triangles}")
    print(f"✅ Saved SBS triangles: {snakemake.output.sbs_triangles}")
    
    # =================================================================
    # STEP 3: TRIANGLE HASH ALIGNMENT (NO ADDITIONAL SCALING)
    # =================================================================
    print("\n=== STEP 3: TRIANGLE HASH ALIGNMENT ===")
    
    # Try triangle hash alignment first (coordinates already scaled)
    print("Attempting triangle hash alignment...")
    try:
        alignment_result = triangle_hash_well_alignment(
            phenotype_positions=phenotype_scaled,  # Use scaled coordinates
            sbs_positions=sbs_positions,
            max_cells_for_hash=75000,
            threshold_triangle=0.1,
            threshold_point=2.0,
            min_score=0.05  # Lower threshold since coordinates are pre-scaled
        )
        
        if not alignment_result.empty:
            best_alignment = alignment_result.iloc[0]
            print(f"✅ Triangle hash alignment successful:")
            print(f"   Score: {best_alignment.get('score', 0):.3f}")
            print(f"   Determinant: {best_alignment.get('determinant', 0):.6f}")
            alignment_approach = "triangle_hash_after_scaling"
        else:
            alignment_result = None
            
    except Exception as e:
        print(f"⚠️  Triangle hash alignment failed: {e}")
        alignment_result = None
    
    # Fallback to identity transformation if triangle hash fails
    if alignment_result is None or alignment_result.empty:
        print("Falling back to identity transformation (no additional changes)...")
        print("Assuming coordinates are already properly aligned after scaling.")
        
        # Validate identity transformation (no translation/rotation)
        from scipy.spatial.distance import cdist
        sample_size = min(10000, len(sbs_positions))
        if len(phenotype_scaled) > sample_size:
            pheno_sample = phenotype_scaled.sample(n=sample_size)[['i', 'j']].values
        else:
            pheno_sample = phenotype_scaled[['i', 'j']].values
            
        if len(sbs_positions) > sample_size:
            sbs_sample = sbs_positions.sample(n=sample_size)[['i', 'j']].values  
        else:
            sbs_sample = sbs_positions[['i', 'j']].values
        
        # No transformation - use coordinates as-is
        distances = cdist(pheno_sample, sbs_sample, metric='euclidean')
        min_distances = distances.min(axis=1)
        score = (min_distances < 10.0).mean()
        
        print(f"Identity transformation validation score: {score:.3f}")
        print(f"Mean nearest neighbor distance: {min_distances.mean():.2f} px")
        
        best_alignment = {
            'rotation': np.eye(2),  # Identity matrix (no rotation)
            'translation': np.array([0.0, 0.0]),  # No translation
            'score': score,
            'determinant': 1.0,
            'transformation_type': 'identity_after_scaling',
            'scale_factor': scale_factor,
            'approach': 'identity_fallback',
            'overlap_fraction': overlap_fraction,
            'validation_mean_distance': min_distances.mean(),
            'validation_median_distance': np.median(min_distances),
            'has_overlap': has_overlap
        }
        alignment_approach = "identity_fallback"
        
        alignment_result = pd.DataFrame([best_alignment])
    
    print(f"✅ Using {alignment_approach} alignment approach")
    
    if alignment_result.empty:
        print("❌ Alignment failed")
        # Create empty outputs
        empty_df = pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
        empty_df.to_parquet(str(snakemake.output.merged_cells))
        
        # ADD THIS: Still save the scaled positions
        phenotype_scaled.to_parquet(str(snakemake.output.phenotype_scaled))
        
        # ADD THIS: Save identity-transformed (since alignment failed)
        phenotype_transformed = phenotype_scaled.copy()
        phenotype_transformed['i_transformed'] = phenotype_scaled['i']
        phenotype_transformed['j_transformed'] = phenotype_scaled['j']
        phenotype_transformed.to_parquet(str(snakemake.output.phenotype_transformed))
        
        # Create empty alignment file with proper structure
        empty_alignment = pd.DataFrame([{
            'rotation_matrix_flat': [1.0, 0.0, 0.0, 1.0],
            'translation_vector': [0.0, 0.0],
            'score': 0.0,
            'determinant': 1.0,
            'transformation_type': 'failed_alignment',
            'scale_factor': float(scale_factor),
            'approach': 'alignment_failed',
            'overlap_fraction': 0.0,
            'validation_mean_distance': 0.0,
            'validation_median_distance': 0.0,
            'has_overlap': False,
            'score_2px': 0.0,
            'score_5px': 0.0,
            'score_10px': 0.0,
            'score_20px': 0.0,
            'score_50px': 0.0
        }])
        empty_alignment.to_parquet(str(snakemake.output.alignment_params))
        
        # Create failure summary
        with open(str(snakemake.output.merge_summary), 'w') as f:
            yaml.dump({'status': 'failed', 'reason': 'alignment_failed'}, f)
        return
    
    best_alignment = alignment_result.iloc[0]
    print(f"✅ Alignment successful:")
    
    # Safe printing with proper handling of missing keys
    alignment_scale_factor = best_alignment.get('scale_factor', scale_factor)
    print(f"   Scale factor: {alignment_scale_factor}")
    print(f"   Score (10px): {best_alignment.get('score', 0):.3f}")
    
    mean_dist = best_alignment.get('validation_mean_distance', 0.0)
    if isinstance(mean_dist, (int, float)) and mean_dist > 0:
        print(f"   Mean distance: {mean_dist:.2f} px")
    else:
        print(f"   Mean distance: N/A")
    
    overlap_frac = best_alignment.get('overlap_fraction', 0.0)
    print(f"   Overlap: {overlap_frac*100:.1f}%")
    
    # Extract only the essential parameters for Parquet serialization with safe conversions
    rotation_matrix = best_alignment.get('rotation', np.eye(2))
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.eye(2)
    else:
        rotation_matrix = np.array(rotation_matrix).reshape(2, 2)
    
    translation_vector = best_alignment.get('translation', np.array([0.0, 0.0]))
    if not isinstance(translation_vector, np.ndarray):
        translation_vector = np.array([0.0, 0.0])
    else:
        translation_vector = np.array(translation_vector).flatten()[:2]
    
    # Apply transformation to scaled phenotype
    pheno_coords_scaled = phenotype_scaled[['i', 'j']].values
    pheno_coords_transformed = pheno_coords_scaled @ rotation_matrix.T + translation_vector
    
    phenotype_transformed = phenotype_scaled.copy()
    phenotype_transformed['i_transformed'] = pheno_coords_transformed[:, 0]
    phenotype_transformed['j_transformed'] = pheno_coords_transformed[:, 1]
    
    # Save transformed positions
    phenotype_transformed.to_parquet(str(snakemake.output.phenotype_transformed))
    print(f"✅ Saved transformed phenotype positions: {snakemake.output.phenotype_transformed}")
    
    # Safe conversion with proper defaults
    def safe_float(value, default=0.0):
        """Safely convert value to float, return default if not possible."""
        try:
            if isinstance(value, str) and value.lower() in ['n/a', 'na', 'none']:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Get scores by threshold safely
    scores_by_threshold = best_alignment.get('scores_by_threshold', {})
    if not isinstance(scores_by_threshold, dict):
        scores_by_threshold = {}
    
    essential_alignment = pd.DataFrame([{
        'rotation_matrix_flat': rotation_matrix.flatten().tolist(),
        'translation_vector': translation_vector.tolist(),
        'scale_factor': safe_float(alignment_scale_factor, scale_factor),
        'score': safe_float(best_alignment.get('score', 0)),
        'determinant': safe_float(best_alignment.get('determinant', 1)),
        'transformation_type': str(best_alignment.get('transformation_type', 'unknown')),
        'approach': str(best_alignment.get('approach', 'unknown')),
        'overlap_fraction': safe_float(best_alignment.get('overlap_fraction', overlap_fraction)),
        'validation_mean_distance': safe_float(best_alignment.get('validation_mean_distance', 0.0)),
        'validation_median_distance': safe_float(best_alignment.get('validation_median_distance', 0.0)),
        'has_overlap': bool(best_alignment.get('has_overlap', has_overlap)),
        # Save individual threshold scores as separate columns
        'score_2px': safe_float(scores_by_threshold.get(2, 0.0)),
        'score_5px': safe_float(scores_by_threshold.get(5, 0.0)),
        'score_10px': safe_float(scores_by_threshold.get(10, 0.0)),
        'score_20px': safe_float(scores_by_threshold.get(20, 0.0)),
        'score_50px': safe_float(scores_by_threshold.get(50, 0.0))
    }])
    
    essential_alignment.to_parquet(str(snakemake.output.alignment_params))
    print(f"✅ Saved alignment parameters: {snakemake.output.alignment_params}")
    
    # =================================================================
    # STEP 4: FINE CELL-LEVEL ALIGNMENT WITH DISTANCE FILTERING
    # =================================================================
    print("\n=== STEP 4: FINE CELL-LEVEL ALIGNMENT ===")
    
    print(f"Using distance threshold: {threshold} pixels")
    
    # Perform cell-level merging using SCALED coordinates
    merged_cells = merge_stitched_cells(
        phenotype_positions=phenotype_scaled,  # Use scaled coordinates!
        sbs_positions=sbs_positions,
        alignment=best_alignment,
        threshold=threshold,
        chunk_size=50000,  # Process in chunks to manage memory
        output_path=str(snakemake.output.merged_cells)  # Convert to string!
    )
    
    if merged_cells.empty:
        print("❌ Cell merging returned no matches")
        empty_df = pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
        empty_df.to_parquet(str(snakemake.output.merged_cells))
        
        # NOTE: phenotype_scaled and phenotype_transformed should already be saved by this point
        # since they're saved in the success path before merging
        
        # Create failure summary
        with open(str(snakemake.output.merge_summary), 'w') as f:
            yaml.dump({'status': 'failed', 'reason': 'no_cell_matches'}, f)
        return
    
    print(f"✅ Cell merging successful:")
    print(f"   Total matches: {len(merged_cells):,}")
    print(f"   Mean distance: {merged_cells['distance'].mean():.2f} px")
    print(f"   Max distance: {merged_cells['distance'].max():.2f} px")
    print(f"   Matches < 5px: {(merged_cells['distance'] < 5).sum():,} ({(merged_cells['distance'] < 5).mean()*100:.1f}%)")
    print(f"   Matches < 10px: {(merged_cells['distance'] < 10).sum():,} ({(merged_cells['distance'] < 10).mean()*100:.1f}%)")
    
    # Add plate and well columns for consistency
    merged_cells['plate'] = plate
    merged_cells['well'] = well
    
    # Reorder columns to match expected format
    output_columns = [
        'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in merged_cells.columns]
    merged_cells_final = merged_cells[available_columns]
    
    # =================================================================
    # SAVE FINAL RESULTS
    # =================================================================
    print("\n=== SAVING FINAL RESULTS ===")
    
    # Save main output
    merged_cells_final.to_parquet(str(snakemake.output.merged_cells))
    print(f"✅ Saved merged cells: {snakemake.output.merged_cells}")
    print(f"   Columns: {list(merged_cells_final.columns)}")
    print(f"   Rows: {len(merged_cells_final):,}")
    
    # Create and save summary statistics
    summary_stats = {
        'status': 'success',
        'plate': plate,
        'well': well,
        'processing_parameters': {
            'scale_factor': float(scale_factor),
            'distance_threshold_pixels': float(threshold)
        },
        'input_data': {
            'phenotype_cells_total': len(phenotype_positions),
            'phenotype_cells_scaled': len(phenotype_scaled),
            'sbs_cells_total': len(sbs_positions),
            'coordinate_overlap_fraction': float(overlap_fraction)
        },
        'step1_coordinate_scaling': {
            'scale_factor': float(scale_factor),
            'phenotype_range_original': {
                'i_min': float(phenotype_positions['i'].min()),
                'i_max': float(phenotype_positions['i'].max()),
                'j_min': float(phenotype_positions['j'].min()),
                'j_max': float(phenotype_positions['j'].max())
            },
            'phenotype_range_scaled': {
                'i_min': float(phenotype_scaled['i'].min()),
                'i_max': float(phenotype_scaled['i'].max()),
                'j_min': float(phenotype_scaled['j'].min()),
                'j_max': float(phenotype_scaled['j'].max())
            },
            'overlap_fraction': float(overlap_fraction)
        },
        'step2_triangle_hashing': {
            'phenotype_triangles': len(phenotype_triangles),
            'sbs_triangles': len(sbs_triangles),
            'coordinates_used': 'scaled_phenotype_and_original_sbs'
        },
        'step3_alignment': {
            'approach_used': alignment_approach,
            'scale_factor_used': safe_float(best_alignment.get('scale_factor', scale_factor)),
            'alignment_score': safe_float(best_alignment.get('score', 0)),
            'determinant': safe_float(best_alignment.get('determinant', 1)),
            'transformation_type': str(best_alignment.get('transformation_type', 'unknown'))
        },
        'step4_cell_merging': {
            'merged_cells_count': len(merged_cells_final),
            'mean_match_distance': float(merged_cells_final['distance'].mean()),
            'max_match_distance': float(merged_cells_final['distance'].max()),
            'matches_under_5px': int((merged_cells_final['distance'] < 5).sum()),
            'matches_under_10px': int((merged_cells_final['distance'] < 10).sum()),
            'match_rate_phenotype': float(len(merged_cells_final) / len(phenotype_positions)),
            'match_rate_sbs': float(len(merged_cells_final) / len(sbs_positions))
        }
    }
    
    # Save summary as separate output
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(summary_stats, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    
    print("\n🎉 === ENHANCED WELL MERGE PIPELINE COMPLETED! ===")
    print(f"Final result: {len(merged_cells_final):,} successfully merged cells")

if __name__ == "__main__":
    main()