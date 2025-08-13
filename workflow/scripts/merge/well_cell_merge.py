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
    validate_matches
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
    # PRELIMINARY DEDUPLICATION (Simple)
    # =================================================================
    print("\n--- Preliminary Deduplication ---")
    
    # Simple deduplication: keep best match per phenotype cell
    # (Full deduplication will happen in Step 3)
    merged_cells = raw_matches.sort_values('distance').drop_duplicates('cell_0', keep='first')
    
    print(f"After simple deduplication: {len(merged_cells):,} matches")
    print(f"Removed: {len(raw_matches) - len(merged_cells):,} duplicate phenotype matches")
    
    # Add plate and well columns
    merged_cells['plate'] = plate
    merged_cells['well'] = well
    
    # Reorder columns
    output_columns = [
        'plate', 'well', 'cell_0', 'i_0', 'j_0', 
        'cell_1', 'i_1', 'j_1', 'distance'
    ]
    
    # Add area columns if available
    if 'area_0' in merged_cells.columns:
        output_columns.insert(-1, 'area_0')
    if 'area_1' in merged_cells.columns:
        output_columns.insert(-1, 'area_1')
    
    available_columns = [col for col in output_columns if col in merged_cells.columns]
    merged_cells_final = merged_cells[available_columns]
    
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
        'summary_stats': summary_stats
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(merge_summary, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 Step 2 (Cell Merge) completed successfully!")
    print(f"Final result: {len(merged_cells_final):,} matched cells")

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