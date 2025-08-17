"""
Step 2: Well Cell Merge - Cell-to-cell matching using alignment parameters.
FIXED VERSION: Properly adds plate/well columns before saving raw matches.
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
    debug_coordinate_uniqueness
)

def main():
    print("=== STEP 2: WELL CELL MERGE (FIXED) ===")
    
    # Load inputs
    phenotype_scaled = validate_dtypes(pd.read_parquet(snakemake.input.scaled_phenotype_positions))
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
            phenotype_positions=phenotype_scaled,
            sbs_positions=sbs_positions,
            alignment=alignment,
            threshold=threshold,
            chunk_size=50000,
            transformed_phenotype_positions=phenotype_transformed
        )
        
        if raw_matches.empty:
            print("❌ No cell matches found")
            create_empty_outputs("no_matches_found")
            return
        
        print(f"✅ Found {len(raw_matches):,} raw matches")
        print(f"Mean distance: {raw_matches['distance'].mean():.2f} px")
        print(f"Max distance: {raw_matches['distance'].max():.2f} px")
        
    except Exception as e:
        print(f"❌ Cell matching failed: {e}")
        import traceback
        traceback.print_exc()
        create_empty_outputs("matching_failed")
        return
    
    # =================================================================
    # ADD PLATE/WELL AND PREPARE OUTPUT (FIXED ORDER)
    # =================================================================
    print("\n--- Preparing Output Data ---")
    
    # FIXED: Add plate and well columns BEFORE saving
    raw_matches['plate'] = plate  
    raw_matches['well'] = well
    
    print(f"Added plate ({plate}) and well ({well}) columns")
    print(f"Raw matches columns: {list(raw_matches.columns)}")
    
    # Add the site information from SBS data
    if 'tile' in sbs_positions.columns:
        # Map each cell_1 back to its tile to get site information
        sbs_tile_map = sbs_positions.set_index('stitched_cell_id')['tile'].to_dict()
        raw_matches['site'] = raw_matches['cell_1'].map(sbs_tile_map)
    else:
        # Fallback if no tile info
        raw_matches['site'] = 0

    # Update column order to include site
    output_columns = [
        'plate', 'well', 'site', 'cell_0', 'i_0', 'j_0', 
        'cell_1', 'i_1', 'j_1', 'distance'
    ]
    
    # Add area columns if available
    if 'area_0' in raw_matches.columns:
        output_columns.insert(-1, 'area_0')
    if 'area_1' in raw_matches.columns:
        output_columns.insert(-1, 'area_1')
    
    # Select available columns
    available_columns = [col for col in output_columns if col in raw_matches.columns]
    raw_matches_ordered = raw_matches[available_columns].copy()
    
    print(f"Final column order: {available_columns}")
    
    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    print("\n--- Saving Outputs ---")
    
    # Save raw matches (now with plate/well columns)
    raw_matches_ordered.to_parquet(str(snakemake.output.raw_matches))
    print(f"✅ Saved raw matches: {snakemake.output.raw_matches}")
    print(f"   Shape: {raw_matches_ordered.shape}")
    print(f"   Columns: {list(raw_matches_ordered.columns)}")
    
    # Save merged cells (same as raw matches in this step)
    raw_matches_ordered.to_parquet(str(snakemake.output.merged_cells))
    print(f"✅ Saved merged cells: {snakemake.output.merged_cells}")
    
    # =================================================================
    # VALIDATE MATCHES
    # =================================================================
    print("\n--- Validating Matches ---")
    
    validation_results = validate_matches(raw_matches_ordered)
    
    print(f"Validation results:")
    print(f"  Status: {validation_results.get('status', 'unknown')}")
    print(f"  Match count: {validation_results.get('match_count', 0):,}")
    if 'distance_stats' in validation_results:
        stats = validation_results['distance_stats']
        print(f"  Mean distance: {stats.get('mean', 0):.2f} px")
        print(f"  Max distance: {stats.get('max', 0):.2f} px")
    
    # =================================================================
    # CREATE SUMMARY
    # =================================================================
    print("\n--- Creating Summary ---")
    
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
            'raw_matches_found': len(raw_matches_ordered),
            'mean_match_distance': float(raw_matches_ordered['distance'].mean()),
            'max_match_distance': float(raw_matches_ordered['distance'].max()),
            'matches_under_5px': int((raw_matches_ordered['distance'] < 5).sum()),
            'matches_under_10px': int((raw_matches_ordered['distance'] < 10).sum()),
            'match_rate_phenotype': float(len(raw_matches_ordered) / len(phenotype_scaled)),
            'match_rate_sbs': float(len(raw_matches_ordered) / len(sbs_positions))
        },
        'validation': validation_results,
        'summary_stats': summary_stats,
        'output_format': {
            'columns_included': available_columns,
            'has_plate_well': 'plate' in available_columns and 'well' in available_columns
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(merge_summary, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 Step 2 (Cell Merge) completed successfully!")
    print(f"Final result: {len(raw_matches_ordered):,} matched cells with plate/well columns")

def create_empty_outputs(reason):
    """Create empty output files when processing fails."""
    # Empty raw matches with proper columns including plate/well
    empty_columns = [
        'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
    ]
    
    empty_matches = pd.DataFrame(columns=empty_columns)
    
    # Add plate and well to empty DataFrame
    empty_matches['plate'] = snakemake.params.plate
    empty_matches['well'] = snakemake.params.well
    
    # Save both outputs
    empty_matches.to_parquet(str(snakemake.output.raw_matches))
    empty_matches.to_parquet(str(snakemake.output.merged_cells))
    
    # Failure summary
    summary = {
        'status': 'failed',
        'reason': reason,
        'plate': snakemake.params.plate,
        'well': snakemake.params.well,
        'output_format': {
            'columns_included': empty_columns,
            'has_plate_well': True
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()