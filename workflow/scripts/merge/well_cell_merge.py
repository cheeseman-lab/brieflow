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
    print("=== STEP 2: WELL CELL MERGE ===")
    
    # Load inputs
    phenotype_scaled = validate_dtypes(pd.read_parquet(snakemake.input.scaled_phenotype_positions))
    phenotype_transformed = validate_dtypes(pd.read_parquet(snakemake.input.transformed_phenotype_positions))
    sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))
    alignment_params = validate_dtypes(pd.read_parquet(snakemake.input.alignment_params))
    
    plate = snakemake.params.plate
    well = snakemake.params.well
    threshold = snakemake.params.threshold
    
    print(f"Processing Plate {plate}, Well {well}")
    print(f"Input: {len(phenotype_scaled):,} phenotype cells, {len(sbs_positions):,} SBS cells")
    print(f"Distance threshold: {threshold} px")
    
    # Verify required ID columns exist
    required_pheno_cols = ['stitched_cell_id', 'original_cell_id']
    required_sbs_cols = ['stitched_cell_id', 'original_cell_id']
    
    missing_pheno = [col for col in required_pheno_cols if col not in phenotype_scaled.columns]
    missing_sbs = [col for col in required_sbs_cols if col not in sbs_positions.columns]
    
    if missing_pheno or missing_sbs:
        print(f"❌ ERROR: Missing required ID columns:")
        print(f"   Phenotype missing: {missing_pheno}")
        print(f"   SBS missing: {missing_sbs}")
        create_empty_outputs("missing_id_columns")
        return
    
    # Load alignment parameters
    print("Loading alignment parameters...")
    
    if len(alignment_params) == 0:
        print("❌ ERROR: No alignment parameters found")
        create_empty_outputs("no_alignment_params")
        return
    
    alignment = load_alignment_parameters(alignment_params.iloc[0])
    
    print(f"Using translation: [{alignment['translation'][0]:.1f}, {alignment['translation'][1]:.1f}], rotation det: {alignment.get('determinant', 1):.3f}")
    print(f"Using {alignment.get('approach', 'unknown')} alignment "
          f"(score: {alignment.get('score', 0):.3f}, "
          f"det: {alignment.get('determinant', 1):.3f})")
    
    # Find cell matches
    print("Finding cell matches...")
    
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
            print("❌ ERROR: No cell matches found")
            create_empty_outputs("no_matches_found")
            return
        
        print(f"Found {len(raw_matches):,} raw matches "
              f"(mean distance: {raw_matches['distance'].mean():.1f}px)")
        
    except Exception as e:
        print(f"❌ ERROR: Cell matching failed: {e}")
        import traceback
        traceback.print_exc()
        create_empty_outputs("matching_failed")
        return
    
    # Map back to original cell IDs
    print("Mapping to original cell IDs...")
    
    # Create mapping dictionaries from stitched → original
    pheno_stitched_to_original = phenotype_scaled.set_index('stitched_cell_id')['original_cell_id'].to_dict()
    sbs_stitched_to_original = sbs_positions.set_index('stitched_cell_id')['original_cell_id'].to_dict()
    
    # Store stitched IDs in separate columns and map to original IDs
    raw_matches['stitched_cell_id_0'] = raw_matches['cell_0']  # Phenotype stitched ID
    raw_matches['stitched_cell_id_1'] = raw_matches['cell_1']  # SBS stitched ID
    
    # Map to original cell IDs
    raw_matches['cell_0'] = raw_matches['stitched_cell_id_0'].map(pheno_stitched_to_original)
    raw_matches['cell_1'] = raw_matches['stitched_cell_id_1'].map(sbs_stitched_to_original)
    
    # Check mapping success
    cell_0_mapped = raw_matches['cell_0'].notna().sum()
    cell_1_mapped = raw_matches['cell_1'].notna().sum()
    
    if cell_0_mapped < len(raw_matches) or cell_1_mapped < len(raw_matches):
        print(f"⚠️  WARNING: Some stitched IDs could not be mapped to original IDs")
        print(f"   Phenotype: {cell_0_mapped}/{len(raw_matches)} mapped")
        print(f"   SBS: {cell_1_mapped}/{len(raw_matches)} mapped")
        
        # Remove unmapped entries
        before_filter = len(raw_matches)
        raw_matches = raw_matches.dropna(subset=['cell_0', 'cell_1'])
        after_filter = len(raw_matches)
        
        if after_filter < before_filter:
            print(f"   Removed {before_filter - after_filter} unmapped matches")
    
    if raw_matches.empty:
        print("❌ ERROR: No matches left after ID mapping")
        create_empty_outputs("mapping_failed")
        return
    
    print(f"Successfully mapped {len(raw_matches):,} matches to original cell IDs")
    
    # Add plate/well and site information
    print("Adding metadata...")
    
    # Add plate and well columns
    raw_matches['plate'] = plate  
    raw_matches['well'] = well
    
    # Map site information using original cell IDs
    site_mapped = False
    if 'tile' in sbs_positions.columns and 'original_cell_id' in sbs_positions.columns:
        sbs_original_to_site = sbs_positions.set_index('original_cell_id')['tile'].to_dict()
        raw_matches['site'] = raw_matches['cell_1'].map(sbs_original_to_site)
        
        mapped_count = raw_matches['site'].notna().sum()
        if mapped_count > 0:
            site_mapped = True
            print(f"Mapped {mapped_count}/{len(raw_matches)} sites from SBS data")
    
    # Fallback to default site
    if not site_mapped:
        raw_matches['site'] = 1
        print("Using default site value: 1")
    
    # Map tile information using original phenotype cell IDs  
    tile_mapped = False
    if 'tile' in phenotype_scaled.columns and 'original_cell_id' in phenotype_scaled.columns:
        pheno_original_to_tile = phenotype_scaled.set_index('original_cell_id')['tile'].to_dict()
        raw_matches['tile'] = raw_matches['cell_0'].map(pheno_original_to_tile)
        
        tile_mapped_count = raw_matches['tile'].notna().sum()
        if tile_mapped_count > 0:
            tile_mapped = True
            print(f"Mapped {tile_mapped_count}/{len(raw_matches)} tiles from phenotype data")
    
    # Fallback to default tile
    if not tile_mapped:
        raw_matches['tile'] = 1
        print("Using default tile value: 1")
    
    # Ensure proper data types
    raw_matches['site'] = raw_matches['site'].astype(int)
    raw_matches['tile'] = raw_matches['tile'].astype(int)
    
    # Prepare final output
    print("Preparing final output...")
    
    # Define output columns with proper order
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
        output_columns.insert(-3, 'area_0')
    if 'area_1' in raw_matches.columns:
        output_columns.insert(-3, 'area_1')
    
    # Select available columns
    available_columns = [col for col in output_columns if col in raw_matches.columns]
    raw_matches_final = raw_matches[available_columns].copy()
    
    # Save outputs
    print("Saving results...")
    
    raw_matches_final.to_parquet(str(snakemake.output.raw_matches))
    raw_matches_final.to_parquet(str(snakemake.output.merged_cells))
    
    # Validate matches
    print("Validating matches...")
    validation_results = validate_matches(raw_matches_final)
    
    if validation_results.get('status') == 'valid':
        if validation_results.get('quality_flags', {}).get('good_quality', False):
            print("✅ Match quality: GOOD")
        else:
            print("⚠️  Match quality: ACCEPTABLE")
            
        # Report key quality metrics
        dist_stats = validation_results.get('distance_stats', {})
        dist_dist = validation_results.get('distance_distribution', {})
        
        print(f"Distance stats: mean={dist_stats.get('mean', 0):.1f}px, "
              f"max={dist_stats.get('max', 0):.1f}px")
        print(f"Quality distribution: "
              f"{dist_dist.get('under_5px', 0)} under 5px, "
              f"{dist_dist.get('under_10px', 0)} under 10px")
              
        # Warn about potential issues
        if validation_results.get('duplication', {}).get('has_duplicates', False):
            pheno_dups = validation_results['duplication']['phenotype_duplicates']
            sbs_dups = validation_results['duplication']['sbs_duplicates']
            print(f"⚠️  WARNING: Found duplicates (phenotype: {pheno_dups}, SBS: {sbs_dups})")
            
        if dist_dist.get('over_20px', 0) > 0:
            print(f"⚠️  WARNING: {dist_dist['over_20px']} matches have distance >20px")
    else:
        print(f"❌ WARNING: Match validation failed: {validation_results.get('status', 'unknown')}")
    
    # Create summary
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
            'phenotype_mapping_success_rate': float(cell_0_mapped / len(raw_matches_final)) if len(raw_matches_final) > 0 else 0.0,
            'sbs_mapping_success_rate': float(cell_1_mapped / len(raw_matches_final)) if len(raw_matches_final) > 0 else 0.0
        },
        'matching_results': {
            'raw_matches_found': len(raw_matches_final),
            'mean_match_distance': float(raw_matches_final['distance'].mean()) if len(raw_matches_final) > 0 else 0.0,
            'max_match_distance': float(raw_matches_final['distance'].max()) if len(raw_matches_final) > 0 else 0.0,
            'matches_under_5px': int((raw_matches_final['distance'] < 5).sum()) if len(raw_matches_final) > 0 else 0,
            'matches_under_10px': int((raw_matches_final['distance'] < 10).sum()) if len(raw_matches_final) > 0 else 0,
            'match_rate_phenotype': float(len(raw_matches_final) / len(phenotype_scaled)) if len(phenotype_scaled) > 0 else 0.0,
            'match_rate_sbs': float(len(raw_matches_final) / len(sbs_positions)) if len(sbs_positions) > 0 else 0.0
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
    
    print(f"✅ Step 2 completed successfully!")
    print(f"Result: {len(raw_matches_final):,} matched cells ready for downstream processing")

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
    
    print(f"❌ Created empty outputs due to: {reason}")

if __name__ == "__main__":
    main()