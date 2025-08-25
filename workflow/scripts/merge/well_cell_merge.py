"""
Step 2: Well Cell Merge - Cell-to-cell matching using alignment parameters.
FIXED VERSION: Operates on stitched_cell_id but preserves original cell_0 for downstream merging.
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
    print("=== STEP 2: WELL CELL MERGE (FIXED FOR ORIGINAL CELL IDS) ===")
    
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
    
    # DEBUG: Check input data structure
    print(f"\nDEBUG - Phenotype columns: {list(phenotype_scaled.columns)}")
    print(f"DEBUG - SBS columns: {list(sbs_positions.columns)}")
    
    # Verify we have the required ID columns
    required_pheno_cols = ['stitched_cell_id', 'original_cell_id']
    required_sbs_cols = ['stitched_cell_id', 'original_cell_id']
    
    missing_pheno = [col for col in required_pheno_cols if col not in phenotype_scaled.columns]
    missing_sbs = [col for col in required_sbs_cols if col not in sbs_positions.columns]
    
    if missing_pheno or missing_sbs:
        print(f"❌ Missing required ID columns:")
        print(f"   Phenotype missing: {missing_pheno}")
        print(f"   SBS missing: {missing_sbs}")
        create_empty_outputs("missing_id_columns")
        return
    
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
    # FIND CELL MATCHES USING STITCHED CELL IDS
    # =================================================================
    print("\n--- Finding Cell Matches (Using Stitched Cell IDs) ---")
    
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
        
        print(f"✅ Found {len(raw_matches):,} raw matches using stitched IDs")
        print(f"Mean distance: {raw_matches['distance'].mean():.2f} px")
        print(f"Max distance: {raw_matches['distance'].max():.2f} px")
        
    except Exception as e:
        print(f"❌ Cell matching failed: {e}")
        import traceback
        traceback.print_exc()
        create_empty_outputs("matching_failed")
        return
    
    # =================================================================
    # CRITICAL FIX: MAP BACK TO ORIGINAL CELL IDS
    # =================================================================
    print("\n--- Mapping to Original Cell IDs ---")
    
    # raw_matches currently has:
    # - cell_0: stitched phenotype cell ID (used for spatial matching)
    # - cell_1: stitched SBS cell ID (used for spatial matching)
    
    # Create mapping dictionaries from stitched → original
    pheno_stitched_to_original = phenotype_scaled.set_index('stitched_cell_id')['original_cell_id'].to_dict()
    sbs_stitched_to_original = sbs_positions.set_index('stitched_cell_id')['original_cell_id'].to_dict()
    
    print(f"Created phenotype mapping: {len(pheno_stitched_to_original):,} stitched → original IDs")
    print(f"Created SBS mapping: {len(sbs_stitched_to_original):,} stitched → original IDs")
    
    # Create new columns with proper naming:
    # - stitched_cell_id_0/1: Keep stitched IDs for validation/debugging
    # - cell_0/1: Map to original IDs for downstream merging
    
    # Store stitched IDs in separate columns
    raw_matches['stitched_cell_id_0'] = raw_matches['cell_0']  # Phenotype stitched ID
    raw_matches['stitched_cell_id_1'] = raw_matches['cell_1']  # SBS stitched ID
    
    # Map to original cell IDs
    raw_matches['cell_0'] = raw_matches['stitched_cell_id_0'].map(pheno_stitched_to_original)
    raw_matches['cell_1'] = raw_matches['stitched_cell_id_1'].map(sbs_stitched_to_original)
    
    # Check mapping success
    cell_0_mapped = raw_matches['cell_0'].notna().sum()
    cell_1_mapped = raw_matches['cell_1'].notna().sum()
    
    print(f"Mapping results:")
    print(f"  Phenotype: {cell_0_mapped}/{len(raw_matches)} mapped to original IDs")
    print(f"  SBS: {cell_1_mapped}/{len(raw_matches)} mapped to original IDs")
    
    if cell_0_mapped < len(raw_matches) or cell_1_mapped < len(raw_matches):
        print(f"⚠️  Warning: Some stitched IDs could not be mapped to original IDs")
        
        # Remove unmapped entries
        before_filter = len(raw_matches)
        raw_matches = raw_matches.dropna(subset=['cell_0', 'cell_1'])
        after_filter = len(raw_matches)
        
        if after_filter < before_filter:
            print(f"   Removed {before_filter - after_filter} unmapped matches")
            print(f"   Continuing with {after_filter:,} successfully mapped matches")
    
    if raw_matches.empty:
        print("❌ No matches left after ID mapping")
        create_empty_outputs("mapping_failed")
        return
    
    print(f"✅ Successfully mapped {len(raw_matches):,} matches to original cell IDs")
    
    # =================================================================
    # ADD PLATE/WELL AND SITE INFORMATION
    # =================================================================
    print("\n--- Adding Plate/Well/Site Information ---")
    
    # Add plate and well columns
    raw_matches['plate'] = plate  
    raw_matches['well'] = well
    
    # Map site information using ORIGINAL cell IDs
    # (since stitched positions might not have site info, but original data should)
    site_mapped = False
    
    # Try to get site from SBS positions using original cell IDs
    if 'tile' in sbs_positions.columns and 'original_cell_id' in sbs_positions.columns:
        print("Mapping site using original cell IDs...")
        sbs_original_to_site = sbs_positions.set_index('original_cell_id')['tile'].to_dict()
        raw_matches['site'] = raw_matches['cell_1'].map(sbs_original_to_site)
        
        mapped_count = raw_matches['site'].notna().sum()
        print(f"   Mapped {mapped_count}/{len(raw_matches)} sites using original cell IDs")
        
        if mapped_count > 0:
            site_mapped = True
    
    # Fallback to default site
    if not site_mapped:
        print("Using fallback site value...")
        raw_matches['site'] = 1
        print(f"   Set all sites to default value: 1")
    
    # Map tile information using ORIGINAL phenotype cell IDs  
    tile_mapped = False
    
    if 'tile' in phenotype_scaled.columns and 'original_cell_id' in phenotype_scaled.columns:
        print("Mapping tile using original phenotype cell IDs...")
        pheno_original_to_tile = phenotype_scaled.set_index('original_cell_id')['tile'].to_dict()
        raw_matches['tile'] = raw_matches['cell_0'].map(pheno_original_to_tile)
        
        tile_mapped_count = raw_matches['tile'].notna().sum()
        print(f"   Mapped {tile_mapped_count}/{len(raw_matches)} tiles using original cell IDs")
        
        if tile_mapped_count > 0:
            tile_mapped = True
    
    # Fallback to default tile
    if not tile_mapped:
        print("Using fallback tile value...")
        raw_matches['tile'] = 1
        print(f"   Set all tiles to default value: 1")
    
    # Ensure proper data types
    raw_matches['site'] = raw_matches['site'].astype(int)
    raw_matches['tile'] = raw_matches['tile'].astype(int)
    
    print(f"Final site distribution: {raw_matches['site'].value_counts().to_dict()}")
    print(f"Final tile distribution: {raw_matches['tile'].value_counts().to_dict()}")
    
    # =================================================================
    # PREPARE FINAL OUTPUT WITH CORRECT COLUMN ORDER
    # =================================================================
    print("\n--- Preparing Final Output ---")
    
    # Define output columns with proper order
    # NOTE: cell_0/1 now contain ORIGINAL cell IDs for downstream merging
    output_columns = [
        'plate', 'well', 'site', 'tile', 
        'cell_0', 'i_0', 'j_0',           # cell_0 = original phenotype cell ID
        'cell_1', 'i_1', 'j_1',           # cell_1 = original SBS cell ID  
        'distance',
        'stitched_cell_id_0',             # Keep stitched IDs for reference
        'stitched_cell_id_1'
    ]
    
    # Add area columns if available
    if 'area_0' in raw_matches.columns:
        output_columns.insert(-3, 'area_0')  # Insert before distance
    if 'area_1' in raw_matches.columns:
        output_columns.insert(-3, 'area_1')  # Insert before distance
    
    # Select available columns
    available_columns = [col for col in output_columns if col in raw_matches.columns]
    raw_matches_final = raw_matches[available_columns].copy()
    
    print(f"Final columns: {available_columns}")
    print(f"Final shape: {raw_matches_final.shape}")
    
    # Verify cell_0 and cell_1 contain original IDs
    print(f"\nVerification - cell_0 (original phenotype IDs) sample: {raw_matches_final['cell_0'].head(3).tolist()}")
    print(f"Verification - cell_1 (original SBS IDs) sample: {raw_matches_final['cell_1'].head(3).tolist()}")
    print(f"Verification - stitched_cell_id_0 sample: {raw_matches_final['stitched_cell_id_0'].head(3).tolist()}")
    print(f"Verification - stitched_cell_id_1 sample: {raw_matches_final['stitched_cell_id_1'].head(3).tolist()}")
    
    # =================================================================
    # SAVE OUTPUTS
    # =================================================================
    print("\n--- Saving Outputs ---")
    
    # Save raw matches with original cell IDs
    raw_matches_final.to_parquet(str(snakemake.output.raw_matches))
    print(f"✅ Saved raw matches: {snakemake.output.raw_matches}")
    
    # Save merged cells (same as raw matches in this step)
    raw_matches_final.to_parquet(str(snakemake.output.merged_cells))
    print(f"✅ Saved merged cells: {snakemake.output.merged_cells}")
    
    # =================================================================
    # VALIDATE MATCHES
    # =================================================================
    print("\n--- Validating Matches ---")
    
    validation_results = validate_matches(raw_matches_final)
    print(f"Validation results: {validation_results.get('status', 'unknown')}")
    
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
        'id_mapping': {
            'stitched_to_original_phenotype': len(pheno_stitched_to_original),
            'stitched_to_original_sbs': len(sbs_stitched_to_original),
            'phenotype_mapping_success_rate': float(cell_0_mapped / len(raw_matches_final)),
            'sbs_mapping_success_rate': float(cell_1_mapped / len(raw_matches_final))
        },
        'matching_results': {
            'raw_matches_found': len(raw_matches_final),
            'mean_match_distance': float(raw_matches_final['distance'].mean()),
            'max_match_distance': float(raw_matches_final['distance'].max()),
            'matches_under_5px': int((raw_matches_final['distance'] < 5).sum()),
            'matches_under_10px': int((raw_matches_final['distance'] < 10).sum()),
            'match_rate_phenotype': float(len(raw_matches_final) / len(phenotype_scaled)),
            'match_rate_sbs': float(len(raw_matches_final) / len(sbs_positions))
        },
        'validation': validation_results,
        'output_format': {
            'columns_included': available_columns,
            'cell_0_contains': 'original_phenotype_cell_ids',
            'cell_1_contains': 'original_sbs_cell_ids',
            'stitched_ids_preserved': 'stitched_cell_id_0' in available_columns and 'stitched_cell_id_1' in available_columns,
            'ready_for_format_merge': True
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(merge_summary, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 Step 2 (Cell Merge) completed successfully!")
    print(f"Final result: {len(raw_matches_final):,} matched cells with ORIGINAL cell IDs")
    print(f"✅ Ready for format_merge: cell_0/cell_1 contain original IDs for downstream merging")

def create_empty_outputs(reason):
    """Create empty output files when processing fails."""
    empty_columns = [
        'plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance', 'stitched_cell_id_0', 'stitched_cell_id_1'
    ]
    
    empty_matches = pd.DataFrame(columns=empty_columns)
    
    # Add default values
    empty_matches['plate'] = snakemake.params.plate
    empty_matches['well'] = snakemake.params.well
    empty_matches['site'] = 1
    empty_matches['tile'] = 1
    
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
            'cell_0_contains': 'original_phenotype_cell_ids',
            'cell_1_contains': 'original_sbs_cell_ids'
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()