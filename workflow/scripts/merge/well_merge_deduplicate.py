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
    # ANALYZE DUPLICATION PATTERNS
    # =================================================================
    print("\n--- Analyzing Duplication Patterns ---")
    
    duplication_analysis = analyze_duplicates(raw_matches)
    
    print(f"Duplication analysis:")
    print(f"  Unique phenotype cells: {duplication_analysis['unique_phenotype_cells']:,}")
    print(f"  Unique SBS cells: {duplication_analysis['unique_sbs_cells']:,}")
    print(f"  Phenotype cells with multiple matches: {duplication_analysis['multi_match_phenotype']:,}")
    print(f"  SBS cells with multiple matches: {duplication_analysis['multi_match_sbs']:,}")
    print(f"  Max matches per phenotype cell: {duplication_analysis['max_phenotype_matches']}")
    print(f"  Max matches per SBS cell: {duplication_analysis['max_sbs_matches']}")
    
    # =================================================================
    # APPLY DEDUPLICATION STRATEGY
    # =================================================================
    print(f"\n--- Applying {dedup_strategy} Deduplication ---")
    
    if dedup_strategy == "greedy_1to1":
        final_matches = greedy_1to1_matching(raw_matches)
        dedup_method = "greedy"
        
    elif dedup_strategy == "hungarian_1to1":
        final_matches = hungarian_1to1_matching(raw_matches)
        dedup_method = "hungarian"
        
    elif dedup_strategy == "simple":
        # Use the already deduplicated matches from Step 2
        final_matches = merged_cells.copy()
        dedup_method = "simple"
        
    else:
        print(f"❌ Unknown deduplication strategy: {dedup_strategy}")
        print("Using simple deduplication as fallback")
        final_matches = merged_cells.copy()
        dedup_method = "simple_fallback"
    
    if final_matches.empty:
        print("❌ Deduplication resulted in no matches")
        create_empty_final_output("deduplication_empty")
        return
    
    print(f"✅ Deduplication complete:")
    print(f"   Final matches: {len(final_matches):,}")
    print(f"   Removed: {len(raw_matches) - len(final_matches):,} duplicates")
    print(f"   Mean distance: {final_matches['distance'].mean():.2f} px")
    print(f"   Max distance: {final_matches['distance'].max():.2f} px")
    
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
    required_columns = ['plate', 'well', 'cell_0', 'i_0', 'j_0', 'cell_1', 'i_1', 'j_1', 'distance']
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
            'method_used': dedup_method,
            'raw_matches_input': len(raw_matches),
            'simple_dedup_input': len(merged_cells),
            'final_matches_output': len(final_output),
            'total_removed': len(raw_matches) - len(final_output),
            'removed_by_advanced_dedup': len(merged_cells) - len(final_output)
        },
        'duplication_analysis': duplication_analysis,
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