"""
Step 2: Well Cell Merge - Cell-to-cell matching using alignment parameters.
Save this as: workflow/scripts/merge/well_cell_merge.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_cell_matching import (
    load_alignment_parameters,
    find_cell_matches,
    validate_matches,
    debug_coordinate_uniqueness  # Import the debug function
)

def main():
    print("=== STEP 2: WELL CELL MERGE ===")
    
    # Load inputs
    phenotype_scaled = validate_dtypes(pd.read_parquet(snakemake.input.scaled_phenotype_positions))  # For metadata
    phenotype_transformed = validate_dtypes(pd.read_parquet(snakemake.input.transformed_phenotype_positions))
    sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))
    alignment_params = validate_dtypes(pd.read_parquet(snakemake.input.alignment_params))
    
    plate = snakemake.params.plate
    well = snakemake.params.well
    threshold = snakemake.params.threshold
    
    print(f"Processing Plate {plate}, Well {well}")
    print(f"Scaled phenotype cells: {len(phenotype_scaled):,}")
    print(f"SBS cells: {len(sbs_positions):,}")
    print(f"Distance threshold: {threshold} px")
    
    # =================================================================
    # DEBUG: INPUT DATA COORDINATE ANALYSIS
    # =================================================================
    print(f"\n🔍 INPUT DATA COORDINATE ANALYSIS")
    
    # Check original phenotype coordinates (before any processing)
    scaled_coords = phenotype_scaled[['i', 'j']].values
    scaled_unique = debug_coordinate_uniqueness(scaled_coords, "Scaled Phenotype Input")
    
    # Check transformed phenotype coordinates
    transformed_coords = phenotype_transformed[['i', 'j']].values
    transformed_unique = debug_coordinate_uniqueness(transformed_coords, "Transformed Phenotype Input")
    
    # Check SBS coordinates
    sbs_coords = sbs_positions[['i', 'j']].values
    sbs_unique = debug_coordinate_uniqueness(sbs_coords, "SBS Input")
    
    print(f"\nInput Data Summary:")
    print(f"  Scaled Phenotype: {scaled_unique:,} unique out of {len(scaled_coords):,} ({100*scaled_unique/len(scaled_coords):.1f}%)")
    print(f"  Transformed Phenotype: {transformed_unique:,} unique out of {len(transformed_coords):,} ({100*transformed_unique/len(transformed_coords):.1f}%)")
    print(f"  SBS: {sbs_unique:,} unique out of {len(sbs_coords):,} ({100*sbs_unique/len(sbs_coords):.1f}%)")
    
    # Check if the issue is already present in input data
    if scaled_unique != len(scaled_coords):
        print("⚠️  ALERT: Scaled phenotype coordinates already have duplicates in input!")
    if transformed_unique != len(transformed_coords):
        print("⚠️  ALERT: Transformed phenotype coordinates already have duplicates in input!")
    if sbs_unique != len(sbs_coords):
        print("⚠️  ALERT: SBS coordinates already have duplicates in input!")
    
    # =================================================================
    # LOAD ALIGNMENT PARAMETERS
    # =================================================================
    print("\n--- Loading Alignment Parameters ---")
    
    if len(alignment_params) == 0:
        print("❌ No alignment parameters found")
        create_empty_outputs("no_alignment_params")
        return
    
    alignment = load_alignment_parameters(alignment_params.iloc[0])
    
    print(f"Alignment approach: {alignment.get('approach', 'unknown')}")
    print(f"Transformation type: {alignment.get('transformation_type', 'unknown')}")
    print(f"Score: {alignment.get('score', 0):.3f}")
    print(f"Determinant: {alignment.get('determinant', 1):.6f}")
    
    # =================================================================
    # FIND CELL MATCHES
    # =================================================================
    print("\n--- Finding Cell Matches ---")
    
    try:
        raw_matches, summary_stats = find_cell_matches(
            phenotype_positions=phenotype_scaled,  # For cell IDs and areas
            sbs_positions=sbs_positions,
            alignment=alignment,
            threshold=threshold,
            chunk_size=50000,
            transformed_phenotype_positions=phenotype_transformed  # For coordinates
        )
        
        if raw_matches.empty:
            print("❌ No cell matches found")
            create_empty_outputs("no_matches_found")
            return
        
        print(f"✅ Found {len(raw_matches):,} raw matches")
        print(f"Mean distance: {raw_matches['distance'].mean():.2f} px")
        print(f"Max distance: {raw_matches['distance'].max():.2f} px")
        
        # =================================================================
        # DEBUG: ANALYZE MATCHING RESULTS
        # =================================================================
        print(f"\n🔍 MATCHING RESULTS ANALYSIS")
        
        # Check coordinate uniqueness in results
        result_pheno_coords = raw_matches[['i_0', 'j_0']].values
        result_sbs_coords = raw_matches[['i_1', 'j_1']].values
        
        result_pheno_unique = debug_coordinate_uniqueness(result_pheno_coords, "Result Phenotype Coordinates")
        result_sbs_unique = debug_coordinate_uniqueness(result_sbs_coords, "Result SBS Coordinates")
        
        print(f"\nMatching Results Coordinate Summary:")
        print(f"  Result Phenotype coords: {result_pheno_unique:,} unique out of {len(result_pheno_coords):,} ({100*result_pheno_unique/len(result_pheno_coords):.1f}%)")
        print(f"  Result SBS coords: {result_sbs_unique:,} unique out of {len(result_sbs_coords):,} ({100*result_sbs_unique/len(result_sbs_coords):.1f}%)")
        
        # Analyze cell ID uniqueness
        pheno_cell_unique = raw_matches['cell_0'].nunique()
        sbs_cell_unique = raw_matches['cell_1'].nunique()
        
        print(f"\nCell ID Uniqueness:")
        print(f"  Unique phenotype cell IDs: {pheno_cell_unique:,} out of {len(raw_matches):,} matches")
        print(f"  Unique SBS cell IDs: {sbs_cell_unique:,} out of {len(raw_matches):,} matches")
        
        # Check if cell IDs vs coordinates mismatch
        if pheno_cell_unique != result_pheno_unique:
            print(f"⚠️  MISMATCH: Phenotype cell IDs ({pheno_cell_unique}) ≠ coordinate pairs ({result_pheno_unique})")
        if sbs_cell_unique != result_sbs_unique:
            print(f"⚠️  MISMATCH: SBS cell IDs ({sbs_cell_unique}) ≠ coordinate pairs ({result_sbs_unique})")
        
        # Analyze top duplicated items
        if pheno_cell_unique < len(raw_matches):
            pheno_duplicates = raw_matches['cell_0'].value_counts()
            print(f"\nTop 5 most duplicated phenotype cells:")
            for i, (cell_id, count) in enumerate(pheno_duplicates.head().items()):
                cell_data = raw_matches[raw_matches['cell_0'] == cell_id].iloc[0]
                print(f"  {i+1}. Cell '{cell_id}': {count} matches at ({cell_data['i_0']:.6f}, {cell_data['j_0']:.6f})")
        
        if sbs_cell_unique < len(raw_matches):
            sbs_duplicates = raw_matches['cell_1'].value_counts()
            print(f"\nTop 5 most duplicated SBS cells:")
            for i, (cell_id, count) in enumerate(sbs_duplicates.head().items()):
                cell_data = raw_matches[raw_matches['cell_1'] == cell_id].iloc[0]
                print(f"  {i+1}. Cell '{cell_id}': {count} matches at ({cell_data['i_1']:.6f}, {cell_data['j_1']:.6f})")
        
        # Compare input vs output uniqueness to see where we lost uniqueness
        print(f"\nUniqueness Loss Analysis:")
        print(f"  Transformed phenotype: {transformed_unique:,} → {pheno_cell_unique:,} (lost {transformed_unique - pheno_cell_unique:,})")
        print(f"  SBS: {sbs_unique:,} → {sbs_cell_unique:,} (lost {sbs_unique - sbs_cell_unique:,})")
        
        # Save raw matches
        raw_matches.to_parquet(str(snakemake.output.raw_matches))
        print(f"✅ Saved raw matches: {snakemake.output.raw_matches}")
        
    except Exception as e:
        print(f"❌ Cell matching failed: {e}")
        import traceback
        traceback.print_exc()
        create_empty_outputs("matching_failed")
        return
    
    # =================================================================
    # SAVE RAW MATCHES (No deduplication here - matches legacy)
    # =================================================================
    print("\n--- Saving Raw Matches ---")

    # Add plate and well columns to raw matches
    raw_matches['plate'] = plate  
    raw_matches['well'] = well

    # Raw matches go directly to output (matching legacy approach)
    merged_cells_final = raw_matches.copy()

    print(f"Raw matches prepared: {len(merged_cells_final):,}")
    print(f"(Deduplication will be handled in Step 3 to match legacy approach)")

    # Reorder columns
    output_columns = [
        'plate', 'well', 'cell_0', 'i_0', 'j_0', 
        'cell_1', 'i_1', 'j_1', 'distance'
    ]
    
    # Add area columns if available
    if 'area_0' in merged_cells_final.columns:
        output_columns.insert(-1, 'area_0')
    if 'area_1' in merged_cells_final.columns:
        output_columns.insert(-1, 'area_1')
    
    available_columns = [col for col in output_columns if col in merged_cells_final.columns]    
    
    # =================================================================
    # VALIDATE MATCHES
    # =================================================================
    print("\n--- Validating Matches ---")
    
    validation_results = validate_matches(merged_cells_final)
    
    print(f"Validation results:")
    for key, value in validation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    print("\n--- Saving Outputs ---")
    
    # Save merged cells
    merged_cells_final.to_parquet(str(snakemake.output.merged_cells))
    print(f"✅ Saved merged cells: {snakemake.output.merged_cells}")
    
    # Create merge summary
    merge_summary = {
        'status': 'success',
        'plate': plate,
        'well': well,
        'processing_parameters': {
            'distance_threshold_pixels': float(threshold)
        },
        'input_data': {
            'scaled_phenotype_cells': len(phenotype_scaled),
            'sbs_cells': len(sbs_positions)
        },
        'alignment_used': {
            'approach': str(alignment.get('approach', 'unknown')),
            'transformation_type': str(alignment.get('transformation_type', 'unknown')),
            'score': float(alignment.get('score', 0)),
            'determinant': float(alignment.get('determinant', 1))
        },
        'matching_results': {
            'raw_matches_found': len(raw_matches),
            'matches_after_simple_dedup': len(merged_cells_final),
            'mean_match_distance': float(merged_cells_final['distance'].mean()),
            'max_match_distance': float(merged_cells_final['distance'].max()),
            'matches_under_5px': int((merged_cells_final['distance'] < 5).sum()),
            'matches_under_10px': int((merged_cells_final['distance'] < 10).sum()),
            'match_rate_phenotype': float(len(merged_cells_final) / len(phenotype_scaled)),
            'match_rate_sbs': float(len(merged_cells_final) / len(sbs_positions))
        },
        'validation': validation_results,
        'summary_stats': summary_stats,
        'debug_analysis': {  # Add debug results to summary
            'input_uniqueness': {
                'scaled_phenotype': scaled_unique,
                'transformed_phenotype': transformed_unique,
                'sbs': sbs_unique
            },
            'result_uniqueness': {
                'phenotype_coordinates': result_pheno_unique,
                'sbs_coordinates': result_sbs_unique,
                'phenotype_cell_ids': pheno_cell_unique,
                'sbs_cell_ids': sbs_cell_unique
            }
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(merge_summary, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 Step 2 (Cell Merge) completed successfully!")
    print(f"Final result: {len(merged_cells_final):,} matched cells")
    
    # =================================================================
    # FINAL DEBUG SUMMARY
    # =================================================================
    print(f"\n🔍 FINAL DEBUG SUMMARY")
    print(f"Input data analysis:")
    print(f"  - Scaled phenotype uniqueness: {100*scaled_unique/len(scaled_coords):.1f}%")
    print(f"  - Transformed phenotype uniqueness: {100*transformed_unique/len(transformed_coords):.1f}%") 
    print(f"  - SBS uniqueness: {100*sbs_unique/len(sbs_coords):.1f}%")
    print(f"Results analysis:")
    print(f"  - Final phenotype cell uniqueness: {100*pheno_cell_unique/len(raw_matches):.1f}%")
    print(f"  - Final SBS cell uniqueness: {100*sbs_cell_unique/len(raw_matches):.1f}%")
    
    if transformed_unique < len(transformed_coords):
        print(f"🎯 ISSUE FOUND: Transformed phenotype coordinates already have duplicates in input data!")
        print(f"   This suggests the coordinate quantization happened in Step 1 (well_alignment.py)")
    elif pheno_cell_unique < transformed_unique:
        print(f"🎯 ISSUE FOUND: Coordinate uniqueness lost during matching process!")
        print(f"   This suggests a bug in the matching logic in Step 2")
    else:
        print(f"🤔 UNEXPECTED: Coordinates appear unique, but validation shows duplicates")
        print(f"   Check cell ID vs coordinate relationship")

def create_empty_outputs(reason):
    """Create empty output files when processing fails."""
    # Empty raw matches
    empty_matches = pd.DataFrame(columns=[
        'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
    ])
    empty_matches.to_parquet(str(snakemake.output.raw_matches))
    
    # Empty merged cells with plate/well
    empty_merged = empty_matches.copy()
    empty_merged['plate'] = snakemake.params.plate
    empty_merged['well'] = snakemake.params.well
    empty_merged.to_parquet(str(snakemake.output.merged_cells))
    
    # Failure summary
    summary = {
        'status': 'failed',
        'reason': reason,
        'plate': snakemake.params.plate,
        'well': snakemake.params.well
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()