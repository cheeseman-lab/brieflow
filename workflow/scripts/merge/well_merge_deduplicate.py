"""
Step 3: Well Merge Deduplication - Advanced deduplication and final processing.
FIXED VERSION: Operates on stitched cell IDs for deduplication but preserves original cell IDs.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_deduplication import (
    greedy_1to1_matching,
    hungarian_1to1_matching,
    analyze_duplicates,
    validate_final_matches
)

def legacy_style_deduplication_stitched_ids(raw_matches):
    """
    Apply legacy deduplication logic but operate on STITCHED cell IDs.
    This ensures proper spatial deduplication while preserving original cell IDs.
    """
    print(f"Applying legacy-compatible deduplication using STITCHED cell IDs")
    print(f"Input: {len(raw_matches):,} raw matches")
    
    # Verify we have the required columns
    required_cols = ['stitched_cell_id_0', 'stitched_cell_id_1', 'cell_0', 'cell_1', 'distance']
    missing_cols = [col for col in required_cols if col not in raw_matches.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns for stitched ID deduplication: {missing_cols}")
        print("Falling back to deduplication on original cell IDs")
        return legacy_style_deduplication_original_ids(raw_matches)
    
    print(f"Operating on:")
    print(f"  - stitched_cell_id_0: phenotype stitched IDs (for deduplication)")
    print(f"  - stitched_cell_id_1: SBS stitched IDs (for deduplication)")
    print(f"  - cell_0/cell_1: original cell IDs (preserved for format_merge)")
    
    # Step 1: For each phenotype STITCHED cell, keep best SBS match
    # Sort by distance to prioritize better matches
    df_sbs_deduped = raw_matches.sort_values('distance', ascending=True).drop_duplicates('stitched_cell_id_0', keep='first')
    
    print(f"After phenotype stitched ID deduplication: {len(df_sbs_deduped):,} matches")
    print(f"Removed: {len(raw_matches) - len(df_sbs_deduped):,} duplicate phenotype stitched matches")
    
    # Step 2: For each remaining SBS STITCHED cell, keep best phenotype match  
    # Sort by distance to prioritize better matches
    df_final = df_sbs_deduped.sort_values('distance', ascending=True).drop_duplicates('stitched_cell_id_1', keep='first')
    
    print(f"After SBS stitched ID deduplication: {len(df_final):,} matches") 
    print(f"Removed: {len(df_sbs_deduped) - len(df_final):,} duplicate SBS stitched matches")
    
    # Verify deduplication worked on stitched IDs
    stitched_pheno_dups = df_final['stitched_cell_id_0'].duplicated().sum()
    stitched_sbs_dups = df_final['stitched_cell_id_1'].duplicated().sum()
    
    print(f"\nDeduplication verification (stitched IDs):")
    print(f"  Phenotype stitched duplicates remaining: {stitched_pheno_dups}")
    print(f"  SBS stitched duplicates remaining: {stitched_sbs_dups}")
    
    if stitched_pheno_dups > 0 or stitched_sbs_dups > 0:
        print(f"⚠️  Warning: Stitched ID deduplication incomplete!")
    else:
        print(f"✅ Perfect stitched ID deduplication achieved")
    
    # Check if original cell IDs still have duplicates (this might be expected)
    original_pheno_dups = df_final['cell_0'].duplicated().sum()
    original_sbs_dups = df_final['cell_1'].duplicated().sum()
    
    print(f"\nOriginal ID status (after stitched deduplication):")
    print(f"  Original phenotype duplicates: {original_pheno_dups}")
    print(f"  Original SBS duplicates: {original_sbs_dups}")
    
    if original_pheno_dups > 0 or original_sbs_dups > 0:
        print(f"  Note: Original ID duplicates are acceptable - one original cell")
        print(f"        may correspond to multiple stitched positions")
    
    return df_final

def legacy_style_deduplication_original_ids(raw_matches):
    """
    Fallback: Apply legacy deduplication on original cell IDs.
    """
    print(f"Applying legacy-compatible deduplication on ORIGINAL cell IDs (fallback)")
    print(f"Input: {len(raw_matches):,} raw matches")
    
    # Step 1: For each phenotype original cell, keep best SBS match
    df_sbs_deduped = raw_matches.sort_values('distance', ascending=True).drop_duplicates('cell_0', keep='first')
    
    print(f"After phenotype original ID deduplication: {len(df_sbs_deduped):,} matches")
    print(f"Removed: {len(raw_matches) - len(df_sbs_deduped):,} duplicate phenotype original matches")
    
    # Step 2: For each remaining SBS original cell, keep best phenotype match  
    df_final = df_sbs_deduped.sort_values('distance', ascending=True).drop_duplicates('cell_1', keep='first')
    
    print(f"After SBS original ID deduplication: {len(df_final):,} matches") 
    print(f"Removed: {len(df_sbs_deduped) - len(df_final):,} duplicate SBS original matches")
    
    return df_final

def main():
    print("=== STEP 3: WELL MERGE DEDUPLICATION (FIXED FOR STITCHED IDS) ===")
    
    # Load inputs
    raw_matches = validate_dtypes(pd.read_parquet(snakemake.input.raw_matches))
    merged_cells = validate_dtypes(pd.read_parquet(snakemake.input.merged_cells))
    
    plate = snakemake.params.plate
    well = snakemake.params.well
    dedup_strategy = snakemake.params.dedup_strategy
    
    print(f"Processing Plate {plate}, Well {well}")
    print(f"Raw matches: {len(raw_matches):,}")
    print(f"Simple deduplicated matches: {len(merged_cells):,}")
    print(f"Deduplication strategy: {dedup_strategy}")
    
    # Debug: Check input columns
    print(f"\nInput columns: {list(raw_matches.columns)}")
    
    if raw_matches.empty:
        print("❌ No raw matches to process")
        create_empty_final_output("no_raw_matches")
        return
    
    # =================================================================
    # APPLY STITCHED-ID-BASED DEDUPLICATION  
    # =================================================================
    print("\n--- Stitched-ID-Based Deduplication ---")
    
    final_matches = legacy_style_deduplication_stitched_ids(raw_matches)
    
    if final_matches.empty:
        print("❌ Deduplication resulted in no matches")
        create_empty_final_output("deduplication_eliminated_all")
        return
    
    # =================================================================
    # VALIDATE FINAL MATCHES
    # =================================================================
    print("\n--- Validating Final Matches ---")
    
    validation_results = validate_final_matches(final_matches)
    
    print(f"Final validation:")
    for key, value in validation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # =================================================================
    # FINAL QUALITY CHECKS
    # =================================================================
    print("\n--- Final Quality Checks ---")
    
    # Check final deduplication status
    if 'stitched_cell_id_0' in final_matches.columns and 'stitched_cell_id_1' in final_matches.columns:
        stitched_pheno_dups = final_matches['stitched_cell_id_0'].duplicated().sum()
        stitched_sbs_dups = final_matches['stitched_cell_id_1'].duplicated().sum()
        print(f"Final stitched ID deduplication:")
        print(f"  Phenotype stitched duplicates: {stitched_pheno_dups}")
        print(f"  SBS stitched duplicates: {stitched_sbs_dups}")
        
        if stitched_pheno_dups == 0 and stitched_sbs_dups == 0:
            print(f"✅ Perfect stitched ID deduplication maintained")
        else:
            print(f"⚠️  Stitched ID duplicates present after final processing")
    
    # Check original cell ID status
    original_pheno_dups = final_matches['cell_0'].duplicated().sum()
    original_sbs_dups = final_matches['cell_1'].duplicated().sum()
    
    print(f"Final original ID status:")
    print(f"  Original phenotype duplicates: {original_pheno_dups}")
    print(f"  Original SBS duplicates: {original_sbs_dups}")
    print(f"  Note: Original ID duplicates are acceptable for format_merge")
    
    # Distance distribution
    distance_stats = {
        'under_1px': (final_matches['distance'] < 1).sum(),
        'under_2px': (final_matches['distance'] < 2).sum(),
        'under_5px': (final_matches['distance'] < 5).sum(),
        'under_10px': (final_matches['distance'] < 10).sum(),
        'over_20px': (final_matches['distance'] > 20).sum(),
    }
    
    print(f"Distance distribution:")
    for threshold, count in distance_stats.items():
        pct = count / len(final_matches) * 100
        print(f"  {threshold}: {count:,} ({pct:.1f}%)")
    
    # Check for unreasonably large distances
    large_distances = final_matches[final_matches['distance'] > 50]
    if len(large_distances) > 0:
        print(f"⚠️  {len(large_distances)} matches have distance > 50px (potential alignment issues)")
        print(f"   Largest distance: {final_matches['distance'].max():.1f} px")
    
    # =================================================================
    # PREPARE FINAL OUTPUT WITH CORRECT FORMAT
    # =================================================================
    print("\n--- Preparing Final Output ---")
    
    # Define proper column order - cell_0/1 must contain ORIGINAL IDs for format_merge
    required_columns = ['plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 'cell_1', 'i_1', 'j_1', 'distance']
    optional_columns = ['area_0', 'area_1', 'stitched_cell_id_0', 'stitched_cell_id_1']
    
    output_columns = [col for col in required_columns if col in final_matches.columns]
    output_columns.extend([col for col in optional_columns if col in final_matches.columns])
    
    final_output = final_matches[output_columns].copy()
    
    # Verify cell_0/cell_1 contain original IDs (critical for format_merge)
    print(f"Verification - cell_0 (original phenotype IDs) sample: {final_output['cell_0'].head(3).tolist()}")
    print(f"Verification - cell_1 (original SBS IDs) sample: {final_output['cell_1'].head(3).tolist()}")
    
    if 'stitched_cell_id_0' in final_output.columns:
        print(f"Verification - stitched_cell_id_0 sample: {final_output['stitched_cell_id_0'].head(3).tolist()}")
    if 'stitched_cell_id_1' in final_output.columns:
        print(f"Verification - stitched_cell_id_1 sample: {final_output['stitched_cell_id_1'].head(3).tolist()}")
    
    # =================================================================
    # SAVE FINAL OUTPUT
    # =================================================================
    print("\n--- Saving Final Output ---")
    
    # Save deduplicated cells (this becomes the main enhanced well merge output)
    final_output.to_parquet(str(snakemake.output.deduplicated_cells))
    print(f"✅ Saved final deduplicated cells: {snakemake.output.deduplicated_cells}")
    print(f"   Shape: {final_output.shape}")
    print(f"   Columns: {output_columns}")
    
    # Create comprehensive deduplication summary
    dedup_summary = {
        'status': 'success',
        'plate': plate,
        'well': well,
        'deduplication': {
            'strategy': dedup_strategy,
            'method_used': 'legacy_on_stitched_ids',
            'raw_matches_input': len(raw_matches),
            'simple_dedup_input': len(merged_cells),
            'final_matches_output': len(final_output),
            'total_removed': len(raw_matches) - len(final_output),
            'removed_by_advanced_dedup': len(merged_cells) - len(final_output)
        },
        'stitched_id_deduplication': {
            'phenotype_stitched_duplicates': int(stitched_pheno_dups) if 'stitched_cell_id_0' in final_matches.columns else 'not_available',
            'sbs_stitched_duplicates': int(stitched_sbs_dups) if 'stitched_cell_id_1' in final_matches.columns else 'not_available',
            'perfect_stitched_dedup': bool(stitched_pheno_dups == 0 and stitched_sbs_dups == 0) if 'stitched_cell_id_0' in final_matches.columns else False
        },
        'original_id_status': {
            'phenotype_original_duplicates': int(original_pheno_dups),
            'sbs_original_duplicates': int(original_sbs_dups),
            'note': 'Original ID duplicates are acceptable - one original cell may map to multiple stitched positions'
        },
        'quality_metrics': {
            'mean_distance': float(final_output['distance'].mean()),
            'median_distance': float(final_output['distance'].median()),
            'max_distance': float(final_output['distance'].max()),
            'std_distance': float(final_output['distance'].std()),
            'distance_distribution': {k: int(v) for k, v in distance_stats.items()}
        },
        'validation': validation_results,
        'final_checks': {
            'large_distance_matches': int(len(large_distances)),
            'largest_distance': float(final_output['distance'].max())
        },
        'output_format': {
            'columns_included': output_columns,
            'cell_0_contains': 'original_phenotype_cell_ids',
            'cell_1_contains': 'original_sbs_cell_ids',
            'stitched_ids_preserved': 'stitched_cell_id_0' in output_columns and 'stitched_cell_id_1' in output_columns,
            'ready_for_format_merge': True
        }
    }
    
    with open(str(snakemake.output.deduplication_summary), 'w') as f:
        yaml.dump(dedup_summary, f, default_flow_style=False)
    
    print(f"✅ Saved deduplication summary: {snakemake.output.deduplication_summary}")
    
    print(f"\n🎉 Step 3 (Deduplication) completed successfully!")
    print(f"Final result: {len(final_output):,} high-quality 1:1 stitched cell matches")
    print(f"✅ Ready for format_merge: cell_0/cell_1 contain ORIGINAL cell IDs")
    print(f"Overall pipeline efficiency: {len(final_output)/len(raw_matches)*100:.1f}% of raw matches retained")

def create_empty_final_output(reason):
    """Create empty final output when deduplication fails."""
    empty_df = pd.DataFrame(columns=[
        'plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance', 'stitched_cell_id_0', 'stitched_cell_id_1'
    ])
    
    # Add plate and well
    empty_df['plate'] = snakemake.params.plate
    empty_df['well'] = snakemake.params.well
    empty_df['site'] = 1
    empty_df['tile'] = 1
    
    empty_df.to_parquet(str(snakemake.output.deduplicated_cells))
    
    # Failure summary
    summary = {
        'status': 'failed',
        'reason': reason,
        'plate': snakemake.params.plate,
        'well': snakemake.params.well,
        'deduplication': {
            'strategy': snakemake.params.dedup_strategy,
            'final_matches_output': 0
        },
        'output_format': {
            'cell_0_contains': 'original_phenotype_cell_ids',
            'cell_1_contains': 'original_sbs_cell_ids'
        }
    }
    
    with open(str(snakemake.output.deduplication_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()