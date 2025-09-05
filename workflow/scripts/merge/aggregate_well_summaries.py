"""Aggregate Well Summaries Script.

Aggregates individual well summary TSV files from the 3-step well merge pipeline into
consolidated plate-level summaries. Converts key-value format summaries (merge/dedup)
to one-row-per-well format for easier analysis.

This script processes:
1. Alignment summaries (already in row format) - from well_alignment rule output [4]
2. Cell merge summaries (converts from key-value to row format) - from well_cell_merge rule output [2]
3. Deduplication summaries (converts from key-value to row format) - from well_merge_deduplicate rule output [1]

Input files (per well):
- alignment_summary.tsv: Well alignment metrics (row format)
- merge_summary.tsv: Cell merge metrics (key-value format)  
- dedup_summary.tsv: Deduplication metrics (key-value format)

Output files (per plate):
- alignment_summaries.tsv: Aggregated alignment data across all wells (output [0])
- cell_merge_summaries.tsv: Aggregated cell merge data across all wells (output [1])
- dedup_summaries.tsv: Aggregated deduplication data across all wells (output [2])

Each output file contains one row per well with plate and well identifier columns.
Failed wells are included with status='failed' and placeholder values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
import re

print("=== AGGREGATE WELL SUMMARIES ===")

plate = snakemake.params.plate
print(f"Processing plate: {plate}")

# Get input file paths - using named inputs from rule
alignment_paths = snakemake.input.alignment_summary_paths
merge_paths = snakemake.input.merge_summary_paths  
dedup_paths = snakemake.input.dedup_summary_paths

print(f"Input files:")
print(f"  Alignment summaries: {len(alignment_paths)}")
print(f"  Merge summaries: {len(merge_paths)}")
print(f"  Deduplication summaries: {len(dedup_paths)}")

def extract_well_id_from_path(file_path: str) -> Tuple[str, str]:
    """Extract plate and well identifiers from file path.
    
    Uses regex patterns to find plate and well identifiers in the file path.
    Handles various naming conventions commonly used in the pipeline.
    
    Args:
        file_path: Path to summary file
        
    Returns:
        Tuple of (plate, well) identifiers
        
    Raises:
        ValueError: If plate/well cannot be extracted from path
    """
    path_str = str(file_path)
    
    # Try different patterns to extract plate and well
    patterns = [
        # Pattern 1: P-123_W-A01 (common in generated filenames)
        r'P-(\d+)_W-([A-H]\d{2})',
        # Pattern 2: plate_123_well_A01
        r'plate[_-](\d+)[_-]well[_-]([A-H]\d{2})',
        # Pattern 3: P123_A01
        r'P(\d+)[_-]([A-H]\d{2})',
        # Pattern 4: just numbers and well format: 123_A01
        r'(\d+)[_-]([A-H]\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path_str, re.IGNORECASE)
        if match:
            plate_id = match.group(1)
            well_id = match.group(2).upper()  # Ensure well is uppercase
            return plate_id, well_id
    
    # If no pattern matches, try to extract from directory structure
    path_parts = Path(file_path).parts
    plate_val, well_val = None, None
    
    for part in path_parts:
        # Look for plate-like patterns
        if plate_val is None:
            plate_match = re.search(r'[Pp]-?(\d+)', part)
            if plate_match:
                plate_val = plate_match.group(1)
        
        # Look for well-like patterns  
        if well_val is None:
            well_match = re.search(r'([A-H]\d{2})', part, re.IGNORECASE)
            if well_match:
                well_val = well_match.group(1).upper()
    
    if plate_val and well_val:
        return plate_val, well_val
    
    raise ValueError(f"Could not extract plate/well from path: {file_path}")

def load_alignment_summary(file_path: str) -> Optional[pd.DataFrame]:
    """Load alignment summary file (already in row format).
    
    Args:
        file_path: Path to alignment summary TSV
        
    Returns:
        DataFrame with alignment summary or None if loading fails
    """
    try:
        if not Path(file_path).exists():
            print(f"⚠️  Alignment summary file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path, sep='\t')
        
        if df.empty:
            print(f"⚠️  Empty alignment summary file: {file_path}")
            return None
        
        # Ensure required columns exist
        if 'plate' not in df.columns or 'well' not in df.columns:
            try:
                plate_id, well_id = extract_well_id_from_path(file_path)
                if 'plate' not in df.columns:
                    df['plate'] = plate_id
                if 'well' not in df.columns:
                    df['well'] = well_id
            except ValueError as e:
                print(f"⚠️  Could not add plate/well to alignment summary: {e}")
                return None
        
        return df
        
    except Exception as e:
        print(f"⚠️  Failed to load alignment summary {file_path}: {e}")
        return None

def load_key_value_summary(file_path: str, summary_type: str) -> Optional[pd.DataFrame]:
    """Load and convert key-value format summary to row format.
    
    Args:
        file_path: Path to summary TSV file
        summary_type: Type of summary ('merge' or 'dedup') for error messages
        
    Returns:
        DataFrame with single row containing all metrics or None if loading fails
    """
    try:
        if not Path(file_path).exists():
            print(f"⚠️  {summary_type} summary file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path, sep='\t')
        
        if df.empty:
            print(f"⚠️  Empty {summary_type} summary file: {file_path}")
            return None
        
        # Expect columns: metric, value
        if 'metric' not in df.columns or 'value' not in df.columns:
            print(f"⚠️  {summary_type} summary {file_path} missing metric/value columns")
            return None
        
        # Convert from key-value to single row
        row_data = {}
        for _, row in df.iterrows():
            metric = str(row['metric'])
            value = row['value']
            
            # Handle different value types
            if pd.isna(value):
                row_data[metric] = None
            elif isinstance(value, str):
                # Try to convert numeric strings
                try:
                    if '.' in value or 'e' in value.lower():
                        row_data[metric] = float(value)
                    else:
                        row_data[metric] = int(value)
                except (ValueError, AttributeError):
                    row_data[metric] = value
            else:
                row_data[metric] = value
        
        # Extract plate/well if not present in the data
        if 'plate' not in row_data or 'well' not in row_data:
            try:
                plate_id, well_id = extract_well_id_from_path(file_path)
                if 'plate' not in row_data:
                    row_data['plate'] = plate_id
                if 'well' not in row_data:
                    row_data['well'] = well_id
            except ValueError as e:
                print(f"⚠️  Could not extract well ID for {summary_type} summary: {e}")
                return None
        
        return pd.DataFrame([row_data])
        
    except Exception as e:
        print(f"⚠️  Failed to load {summary_type} summary {file_path}: {e}")
        return None

def create_failed_well_placeholder(plate_id: str, well_id: str, summary_type: str) -> pd.DataFrame:
    """Create placeholder row for failed wells.
    
    Args:
        plate_id: Plate identifier
        well_id: Well identifier
        summary_type: Type of summary for appropriate default columns
        
    Returns:
        DataFrame with single placeholder row
    """
    base_data = {
        'plate': plate_id,
        'well': well_id,
        'status': 'failed',
    }
    
    if summary_type == 'alignment':
        # Add alignment-specific placeholder columns
        placeholder_data = {
            **base_data,
            'failure_reason': 'file_missing',
            'scale_factor': np.nan,
            'overlap_fraction': np.nan,
            'phenotype_triangles': 0,
            'sbs_triangles': 0,
            'alignment_score': np.nan,
            'determinant': np.nan,
            'approach': 'failed',
            'transformation_type': 'failed',
        }
    elif summary_type == 'merge':
        # Add merge-specific placeholder columns  
        placeholder_data = {
            **base_data,
            'reason': 'file_missing',
            'distance_threshold_pixels': np.nan,
            'phenotype_cells_before_filtering': 0,
            'sbs_cells_before_filtering': 0,
            'raw_matches_found': 0,
            'mean_match_distance': np.nan,
        }
    elif summary_type == 'dedup':
        # Add deduplication-specific placeholder columns
        placeholder_data = {
            **base_data,
            'error': 'file_missing',
            'processing_final_matches_output': 0,
            'deduplication_achieved_1to1_stitched': False,
            'quality_match_count': 0,
            'quality_mean_distance': np.nan,
        }
    else:
        placeholder_data = base_data
    
    return pd.DataFrame([placeholder_data])

def get_all_expected_wells(file_paths: List[str]) -> List[Tuple[str, str]]:
    """Extract all expected well identifiers from file paths.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of (plate, well) tuples for all expected wells
    """
    wells = []
    for path in file_paths:
        try:
            plate_id, well_id = extract_well_id_from_path(path)
            wells.append((plate_id, well_id))
        except ValueError:
            print(f"⚠️  Could not extract well ID from path: {path}")
            continue
    
    return sorted(set(wells))  # Remove duplicates and sort

def aggregate_summaries(file_paths: List[str], summary_type: str) -> pd.DataFrame:
    """Aggregate summary files into a single DataFrame.
    
    Args:
        file_paths: List of paths to summary files
        summary_type: Type of summary ('alignment', 'merge', or 'dedup')
        
    Returns:
        DataFrame with aggregated summaries (one row per well)
    """
    print(f"Aggregating {len(file_paths)} {summary_type} summary files...")
    
    if not file_paths:
        print(f"⚠️  No {summary_type} summary files provided")
        return pd.DataFrame()
    
    all_wells = get_all_expected_wells(file_paths)
    print(f"Expected wells: {len(all_wells)}")
    
    aggregated_rows = []
    processed_wells = set()
    
    # Process existing files
    for file_path in file_paths:
        try:
            plate_id, well_id = extract_well_id_from_path(file_path)
            well_key = (plate_id, well_id)
            
            if summary_type == 'alignment':
                df = load_alignment_summary(file_path)
            else:
                df = load_key_value_summary(file_path, summary_type)
            
            if df is not None and not df.empty:
                # Ensure plate/well columns are present and correct
                df['plate'] = plate_id
                df['well'] = well_id
                aggregated_rows.append(df)
                processed_wells.add(well_key)
                print(f"✅ Processed {summary_type} summary for {plate_id}-{well_id}")
            else:
                print(f"⚠️  Empty or invalid {summary_type} summary for {plate_id}-{well_id}")
                # Create placeholder for invalid file
                placeholder = create_failed_well_placeholder(plate_id, well_id, summary_type)
                aggregated_rows.append(placeholder)
                processed_wells.add(well_key)
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Create placeholders for missing wells
    missing_wells = set(all_wells) - processed_wells
    for plate_id, well_id in missing_wells:
        print(f"⚠️  Creating placeholder for missing {summary_type} summary: {plate_id}-{well_id}")
        placeholder = create_failed_well_placeholder(plate_id, well_id, summary_type)
        aggregated_rows.append(placeholder)
    
    # Combine all rows
    if aggregated_rows:
        result = pd.concat(aggregated_rows, ignore_index=True)
        
        # Sort by plate and well for consistent output
        result = result.sort_values(['plate', 'well']).reset_index(drop=True)
        
        print(f"✅ Aggregated {len(result)} {summary_type} summaries")
        return result
    else:
        print(f"❌ No {summary_type} summaries could be processed")
        return pd.DataFrame()

# Main aggregation logic
try:
    # Process alignment summaries (already in row format)
    print("\n--- Processing Alignment Summaries ---")
    alignment_df = aggregate_summaries(alignment_paths, 'alignment')
    
    # Process merge summaries (convert from key-value to row format)
    print("\n--- Processing Cell Merge Summaries ---")
    merge_df = aggregate_summaries(merge_paths, 'merge')
    
    # Process deduplication summaries (convert from key-value to row format)  
    print("\n--- Processing Deduplication Summaries ---")
    dedup_df = aggregate_summaries(dedup_paths, 'dedup')
    
    # Save aggregated summaries
    print("\n--- Saving Aggregated Summaries ---")
    
    # Create output directories if needed
    for output_path in [snakemake.output.alignment_summaries, 
                      snakemake.output.cell_merge_summaries,
                      snakemake.output.dedup_summaries]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save alignment summaries
    if not alignment_df.empty:
        alignment_df.to_csv(snakemake.output.alignment_summaries, sep='\t', index=False)
        print(f"✅ Saved alignment summaries: {snakemake.output.alignment_summaries}")
        print(f"   {len(alignment_df)} wells processed")
    else:
        # Create empty file with headers
        pd.DataFrame(columns=['plate', 'well', 'status']).to_csv(
            snakemake.output.alignment_summaries, sep='\t', index=False)
        print(f"⚠️  Saved empty alignment summaries: {snakemake.output.alignment_summaries}")
    
    # Save cell merge summaries  
    if not merge_df.empty:
        merge_df.to_csv(snakemake.output.cell_merge_summaries, sep='\t', index=False)
        print(f"✅ Saved cell merge summaries: {snakemake.output.cell_merge_summaries}")
        print(f"   {len(merge_df)} wells processed")
    else:
        pd.DataFrame(columns=['plate', 'well', 'status']).to_csv(
            snakemake.output.cell_merge_summaries, sep='\t', index=False)
        print(f"⚠️  Saved empty cell merge summaries: {snakemake.output.cell_merge_summaries}")
    
    # Save deduplication summaries
    if not dedup_df.empty:
        dedup_df.to_csv(snakemake.output.dedup_summaries, sep='\t', index=False)
        print(f"✅ Saved deduplication summaries: {snakemake.output.dedup_summaries}")  
        print(f"   {len(dedup_df)} wells processed")
    else:
        pd.DataFrame(columns=['plate', 'well', 'status']).to_csv(
            snakemake.output.dedup_summaries, sep='\t', index=False)
        print(f"⚠️  Saved empty deduplication summaries: {snakemake.output.dedup_summaries}")
    
    print(f"\n🎉 Successfully aggregated summaries for plate {plate}")
    
    # Print summary statistics
    successful_wells = []
    if not alignment_df.empty:
        successful_alignment = len(alignment_df[alignment_df.get('status', '') != 'failed'])
        successful_wells.append(f"Alignment: {successful_alignment}/{len(alignment_df)}")
    
    if not merge_df.empty:
        successful_merge = len(merge_df[merge_df.get('status', '') != 'failed'])  
        successful_wells.append(f"Merge: {successful_merge}/{len(merge_df)}")
        
    if not dedup_df.empty:
        successful_dedup = len(dedup_df[dedup_df.get('status', '') != 'failed'])
        successful_wells.append(f"Dedup: {successful_dedup}/{len(dedup_df)}")
    
    if successful_wells:
        print(f"Success rates: {', '.join(successful_wells)}")
    
except Exception as e:
    print(f"❌ Error during aggregation: {e}")
    import traceback
    traceback.print_exc()
    raise

print("=== AGGREGATE WELL SUMMARIES COMPLETED ===")