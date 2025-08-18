"""
Step 3: Well Merge Deduplication - Advanced deduplication and final processing.
Save this as: workflow/scripts/merge/well_merge_deduplicate.py
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

def legacy_style_deduplication(raw_matches):
    """
    Apply the exact legacy deduplication logic from deduplicate_cells function.
    Two-step process matching the original legacy implementation.
    """
    print(f"Applying legacy-compatible deduplication")
    print(f"Input: {len(raw_matches):,} raw matches")
    
    # Step 1: For each phenotype cell, keep best SBS match
    # Sort by distance to prioritize better matches (equivalent to legacy sorting logic)
    df_sbs_deduped = raw_matches.sort_values('distance', ascending=True).drop_duplicates('cell_0', keep='first')
    
    print(f"After phenotype deduplication: {len(df_sbs_deduped):,} matches")
    print(f"Removed: {len(raw_matches) - len(df_sbs_deduped):,} duplicate phenotype matches")
    
    # Step 2: For each remaining SBS cell, keep best phenotype match  
    # Sort by distance to prioritize better matches
    df_final = df_sbs_deduped.sort_values('distance', ascending=True).drop_duplicates('cell_1', keep='first')
    
    print(f"After SBS deduplication: {len(df_final):,} matches") 
    print(f"Removed: {len(df_sbs_deduped) - len(df_final):,} duplicate SBS matches")
    
    # Legacy statistics
    print(f"\nLegacy-compatible deduplication complete:")
    print(f"  Initial cells: {len(raw_matches):,}")
    print(f"  After phenotype dedup: {len(df_sbs_deduped):,}")
    print(f"  After SBS dedup: {len(df_final):,}")
    
    return df_final

def main():
    print("=== STEP 3: WELL MERGE DEDUPLICATION ===")
    
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
    
    if raw_matches.empty:
        print("❌ No raw matches to process")
        create_empty_final_output("no_raw_matches")
        return
    
    # =================================================================
    # APPLY LEGACY-STYLE DEDUPLICATION  
    # =================================================================
    print("\n--- Legacy-Style Deduplication ---")
    
    final_matches = legacy_style_deduplication(raw_matches)
    
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
    
    # Check for any remaining duplicates
    pheno_dups = final_matches['cell_0'].duplicated().sum()
    sbs_dups = final_matches['cell_1'].duplicated().sum()
    
    if pheno_dups > 0 or sbs_dups > 0:
        print(f"⚠️  WARNING: Duplicates still present!")
        print(f"   Phenotype duplicates: {pheno_dups}")
        print(f"   SBS duplicates: {sbs_dups}")
    else:
        print(f"✅ No duplicates remaining - true 1:1 matching achieved")
    
    # =================================================================
    # FINAL QUALITY CHECKS
    # =================================================================
    print("\n--- Final Quality Checks ---")
    
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
    # SAVE FINAL OUTPUT
    # =================================================================
    print("\n--- Saving Final Output ---")
    
    # Ensure proper column order and format
    required_columns = ['plate', 'well', 'site', 'tile', 'cell_0', 'i_0', 'j_0', 'cell_1', 'i_1', 'j_1', 'distance']
    optional_columns = ['area_0', 'area_1']
    
    output_columns = [col for col in required_columns if col in final_matches.columns]
    output_columns.extend([col for col in optional_columns if col in final_matches.columns])
    
    final_output = final_matches[output_columns].copy()
    
    # Save deduplicated cells (this becomes the main enhanced well merge output)
    final_output.to_parquet(str(snakemake.output.deduplicated_cells))
    print(f"✅ Saved final deduplicated cells: {snakemake.output.deduplicated_cells}")
    
    # Create comprehensive deduplication summary
    dedup_summary = {
        'status': 'success',
        'plate': plate,
        'well': well,
        'deduplication': {
            'strategy': dedup_strategy,
            'method_used': 'legacy',
            'raw_matches_input': len(raw_matches),
            'simple_dedup_input': len(merged_cells),
            'final_matches_output': len(final_output),
            'total_removed': len(raw_matches) - len(final_output),
            'removed_by_advanced_dedup': len(merged_cells) - len(final_output)
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
            'phenotype_duplicates': int(pheno_dups),
            'sbs_duplicates': int(sbs_dups),
            'is_true_1to1': bool(pheno_dups == 0 and sbs_dups == 0),
            'large_distance_matches': int(len(large_distances)),
            'largest_distance': float(final_output['distance'].max())
        }
    }
    
    with open(str(snakemake.output.deduplication_summary), 'w') as f:
        yaml.dump(dedup_summary, f, default_flow_style=False)
    
    print(f"✅ Saved deduplication summary: {snakemake.output.deduplication_summary}")
    
    print(f"\n🎉 Step 3 (Deduplication) completed successfully!")
    print(f"Final result: {len(final_output):,} high-quality 1:1 cell matches")
    print(f"Overall pipeline efficiency: {len(final_output)/len(raw_matches)*100:.1f}% of raw matches retained")

def create_empty_final_output(reason):
    """Create empty final output when deduplication fails."""
    empty_df = pd.DataFrame(columns=[
        'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
        'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
    ])
    
    # Add plate and well
    empty_df['plate'] = snakemake.params.plate
    empty_df['well'] = snakemake.params.well
    
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
        }
    }
    
    with open(str(snakemake.output.deduplication_summary), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    main()