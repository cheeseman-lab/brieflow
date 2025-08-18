"""
Step 2: Well Cell Merge - Cell-to-cell matching using alignment parameters.
FIXED VERSION: Properly handles site column creation and debugging.
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
    
    # DEBUG: Check SBS positions structure
    print(f"\nDEBUG - SBS positions columns: {list(sbs_positions.columns)}")
    print(f"DEBUG - SBS positions sample:")
    print(sbs_positions.head(2))
    
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
    # ADD PLATE/WELL AND SITE INFORMATION (FIXED)
    # =================================================================
    print("\n--- Preparing Output Data ---")
    
    # Add plate and well columns
    raw_matches['plate'] = plate  
    raw_matches['well'] = well
    
    print(f"Added plate ({plate}) and well ({well}) columns")
    
    # FIXED: Better site handling with detailed debugging
    print(f"\nDEBUG - Raw matches columns before site: {list(raw_matches.columns)}")
    print(f"DEBUG - Raw matches cell_1 sample: {raw_matches['cell_1'].head(3).tolist()}")
    
    # Check what identifier columns exist in SBS positions
    potential_id_cols = [col for col in sbs_positions.columns if 'cell' in col.lower() or 'id' in col.lower()]
    print(f"DEBUG - Potential SBS ID columns: {potential_id_cols}")
    
    # Try multiple approaches to map site information
    site_mapped = False
    
    # Approach 1: Use stitched_cell_id if available
    if 'stitched_cell_id' in sbs_positions.columns:
        print("Trying to map site using stitched_cell_id...")
        
        # Check if tile column exists
        if 'tile' in sbs_positions.columns:
            sbs_site_map = sbs_positions.set_index('stitched_cell_id')['tile'].to_dict()
            raw_matches['site'] = raw_matches['cell_1'].map(sbs_site_map)
            
            # Check success rate
            mapped_count = raw_matches['site'].notna().sum()
            print(f"   Mapped {mapped_count}/{len(raw_matches)} sites using tile column")
            
            if mapped_count > 0:
                site_mapped = True
            else:
                print("   No sites mapped via stitched_cell_id -> tile")
        else:
            print("   No 'tile' column in SBS positions")
    
    # Approach 2: Try using other ID columns
    if not site_mapped:
        print("Trying alternative site mapping approaches...")
        
        # Look for other potential site/tile columns
        site_cols = [col for col in sbs_positions.columns if col in ['site', 'tile_id', 'tile_index']]
        if site_cols:
            site_col = site_cols[0]
            print(f"   Found potential site column: {site_col}")
            
            if 'stitched_cell_id' in sbs_positions.columns:
                sbs_site_map = sbs_positions.set_index('stitched_cell_id')[site_col].to_dict()
                raw_matches['site'] = raw_matches['cell_1'].map(sbs_site_map)
                
                mapped_count = raw_matches['site'].notna().sum()
                print(f"   Mapped {mapped_count}/{len(raw_matches)} sites using {site_col}")
                
                if mapped_count > 0:
                    site_mapped = True
    
    # Approach 3: Extract from cell_1 if it contains site info
    if not site_mapped:
        print("Trying to extract site from cell_1 identifier...")
        
        # Check if cell_1 has a pattern that includes site/tile info
        sample_cell_1 = raw_matches['cell_1'].iloc[0] if len(raw_matches) > 0 else ""
        print(f"   Sample cell_1: {sample_cell_1}")
        
        # If cell_1 looks like it has site info (e.g., "plate_well_site_cell")
        if isinstance(sample_cell_1, str) and len(sample_cell_1.split('_')) >= 4:
            try:
                raw_matches['site'] = raw_matches['cell_1'].str.split('_').str[2].astype(int)
                print(f"   Extracted site from cell_1 identifier pattern")
                site_mapped = True
            except:
                print("   Failed to extract site from cell_1 pattern")
    
    # Fallback: Use default site
    if not site_mapped:
        print("Using fallback site value...")
        raw_matches['site'] = 1  # Default site value
        print(f"   Set all sites to default value: 1")
    
    # Verify site column
    print(f"Final site distribution: {raw_matches['site'].value_counts().to_dict()}")
    print(f"Site column type: {raw_matches['site'].dtype}")
    
    # Convert site to int if it's not already
    if raw_matches['site'].dtype != 'int64':
        try:
            raw_matches['site'] = raw_matches['site'].astype(int)
            print("Converted site column to int")
        except:
            print("Warning: Could not convert site to int, keeping as-is")
    
    # =================================================================
    # ADD TILE INFORMATION (FROM PHENOTYPE DATA)
    # =================================================================
    print(f"\nAdding tile information from phenotype data...")
    
    # Check what identifier columns exist in phenotype positions
    potential_pheno_id_cols = [col for col in phenotype_scaled.columns if 'cell' in col.lower() or 'id' in col.lower()]
    print(f"DEBUG - Potential phenotype ID columns: {potential_pheno_id_cols}")
    
    tile_mapped = False
    
    # Try to map tile information from phenotype data
    if 'stitched_cell_id' in phenotype_scaled.columns and 'tile' in phenotype_scaled.columns:
        print("Mapping tile using stitched_cell_id from phenotype...")
        
        pheno_tile_map = phenotype_scaled.set_index('stitched_cell_id')['tile'].to_dict()
        raw_matches['tile'] = raw_matches['cell_0'].map(pheno_tile_map)
        
        # Check success rate
        tile_mapped_count = raw_matches['tile'].notna().sum()
        print(f"   Mapped {tile_mapped_count}/{len(raw_matches)} tiles using phenotype tile column")
        
        if tile_mapped_count > 0:
            tile_mapped = True
        else:
            print("   No tiles mapped via phenotype stitched_cell_id -> tile")
    else:
        print("   Missing stitched_cell_id or tile column in phenotype positions")
    
    # Fallback: Use default tile
    if not tile_mapped:
        print("Using fallback tile value...")
        raw_matches['tile'] = 1  # Default tile value
        print(f"   Set all tiles to default value: 1")
    
    # Verify tile column
    print(f"Final tile distribution: {raw_matches['tile'].value_counts().to_dict()}")
    print(f"Tile column type: {raw_matches['tile'].dtype}")
    
    # Convert tile to int if it's not already
    if raw_matches['tile'].dtype != 'int64':
        try:
            raw_matches['tile'] = raw_matches['tile'].astype(int)
            print("Converted tile column to int")
        except:
            print("Warning: Could not convert tile to int, keeping as-is")
    
    print(f"Raw matches columns after site and tile: {list(raw_matches.columns)}")
    
    # Update column order to include site and tile
    output_columns = [
        'plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 
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
    
    # Save raw matches (now with plate/well/site columns)
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
            'has_plate_well': 'plate' in available_columns and 'well' in available_columns,
            'has_site': 'site' in available_columns,
            'has_tile': 'tile' in available_columns,
            'site_mapping_method': 'stitched_cell_id' if site_mapped else 'fallback',
            'tile_mapping_method': 'stitched_cell_id' if tile_mapped else 'fallback'
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(merge_summary, f, default_flow_style=False)
    
    print(f"✅ Saved merge summary: {snakemake.output.merge_summary}")
    print(f"\n🎉 Step 2 (Cell Merge) completed successfully!")
    print(f"Final result: {len(raw_matches_ordered):,} matched cells with plate/well/site columns")

def create_empty_outputs(reason):
    """Create empty output files when processing fails."""
    # Empty raw matches with proper columns including plate/well/site/tile
    empty_columns = [
        'plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
    ]
    
    empty_matches = pd.DataFrame(columns=empty_columns)
    
    # Add plate, well, site, and tile to empty DataFrame
    empty_matches['plate'] = snakemake.params.plate
    empty_matches['well'] = snakemake.params.well
    empty_matches['site'] = 1  # Default site
    empty_matches['tile'] = 1  # Default tile
    
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
            'has_plate_well': True,
            'has_site': True,
            'has_tile': True
        }
    }
    
    with open(str(snakemake.output.merge_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()