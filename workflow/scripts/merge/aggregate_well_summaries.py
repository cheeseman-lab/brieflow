"""Aggregate Well Summaries Script.

Aggregates individual well summary TSV files from the 3-step well merge pipeline into
consolidated plate-level summaries. Converts key-value format summaries (merge/dedup)
to one-row-per-well format for easier analysis.

This script processes:
1. Alignment summaries (already in row format) 
2. Cell merge summaries (converts from key-value to row format)
3. Deduplication summaries (converts from key-value to row format)

Input files (per well):
- alignment_summary.tsv: Well alignment metrics (row format)
- merge_summary.tsv: Cell merge metrics (key-value format)  
- dedup_summary.tsv: Deduplication metrics (key-value format)

Output files (per plate):
- alignment_summaries.tsv: Aggregated alignment data across all wells
- cell_merge_summaries.tsv: Aggregated cell merge data across all wells  
- dedup_summaries.tsv: Aggregated deduplication data across all wells

Each output file contains one row per well with plate and well identifier columns.
Failed wells are included with status='failed' and placeholder values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

print("=== AGGREGATE WELL SUMMARIES ===")

plate = snakemake.params.plate
print(f"Processing plate: {plate}")

# Get input file paths
alignment_paths = snakemake.input.alignment_summary_paths
merge_paths = snakemake.input.merge_summary_paths  
dedup_paths = snakemake.input.dedup_summary_paths

print(f"Input files:")
print(f"  Alignment summaries: {len(alignment_paths)}")
print(f"  Merge summaries: {len(merge_paths)}")
print(f"  Deduplication summaries: {len(dedup_paths)}")

# Extract plate and well identifiers from file path
def extract_well_id_from_path(file_path: str) -> tuple[str, str]:
    """Extract plate and well identifiers from file path.
    
    Assumes file paths follow the pattern containing plate and well identifiers.
    
    Args:
        file_path: Path to summary file
        
    Returns:
        Tuple of (plate, well) identifiers
        
    Raises:
        ValueError: If plate/well cannot be extracted from path
    """
    path = Path(file_path)
    
    # Try to extract from filename patterns like: plate_P001_well_A01_summary.tsv
    filename = path.stem
    parts = filename.split('_')
    
    plate, well = None, None
    
    # Look for plate and well in filename parts
    for i, part in enumerate(parts):
        if part == 'plate' and i + 1 < len(parts):
            plate = parts[i + 1]
        elif part == 'well' and i + 1 < len(parts):
            well = parts[i + 1]
    
    if plate is None or well is None:
        # Fallback: try to extract from directory structure or other patterns
        # This is a backup strategy - adjust based on your actual file naming
        raise ValueError(f"Could not extract plate/well from path: {file_path}")
    
    return plate, well

# Load alignment summary file (already in row format)
def load_alignment_summary(file_path: str) -> Optional[pd.DataFrame]:
    """Load alignment summary file (already in row format).
    
    Args:
        file_path: Path to alignment summary TSV
        
    Returns:
        DataFrame with alignment summary or None if loading fails
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Ensure required columns exist
        if 'plate' not in df.columns or 'well' not in df.columns:
            plate, well = extract_well_id_from_path(file_path)
            if 'plate' not in df.columns:
                df['plate'] = plate
            if 'well' not in df.columns:
                df['well'] = well
        
        return df
        
    except Exception as e:
        print(f"⚠️  Failed to load alignment summary {file_path}: {e}")
        return None

# Load and convert key-value format summary to row format
def load_key_value_summary(file_path: str, summary_type: str) -> Optional[pd.DataFrame]:
    """Load and convert key-value format summary to row format.
    
    Args:
        file_path: Path to summary TSV file
        summary_type: Type of summary ('merge' or 'dedup') for error messages
        
    Returns:
        DataFrame with single row containing all metrics or None if loading fails
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        
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
                    if '.' in value:
                        row_data[metric] = float(value)
                    else:
                        row_data[metric] = int(value)
                except ValueError:
                    row_data[metric] = value
            else:
                row_data[metric] = value
        
        # Extract plate/well if not present in the data
        if 'plate' not in row_data or 'well' not in row_data:
            try:
                plate, well = extract_well_id_from_path(file_path)
                if 'plate' not in row_data:
                    row_data['plate'] = plate
                if 'well' not in row_data:
                    row_data['well'] = well
            except ValueError as e:
                print(f"⚠️  Could not extract well ID for {summary_type} summary: {e}")
                return None
        
        return pd.DataFrame([row_data])
        
    except Exception as e:
        print(f"⚠️  Failed to load {summary_type} summary {file_path}: {e}")
        return None

# Create placeholder row for failed wells
def create_failed_well_placeholder(plate: str, well: str, summary_type: str) -> pd.DataFrame:
    """Create placeholder row for failed wells.
    
    Args:
        plate: Plate identifier
        well: Well identifier
        summary_type: Type of summary for appropriate default columns
        
    Returns:
        DataFrame with single placeholder row
    """
    base_data = {
        'plate': plate,
        'well': well,
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

# Extract all expected well identifiers from file paths
def get_all_expected_wells(file_paths: List[str]) -> List[tuple[str, str]]:
    """Extract all expected well identifiers from file paths.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of (plate, well) tuples for all expected wells
    """
    wells = []
    for path in file_paths:
        try:
            plate, well = extract_well_id_from_path(path)
            wells.append((plate, well))
        except ValueError:
            print(f"⚠️  Could not extract well ID from path: {path}")
            continue
    
    return sorted(set(wells))  # Remove duplicates and sort

# Aggregate summary files into a single DataFrame
def aggregate_summaries(file_paths: List[str], summary_type: str) -> pd.DataFrame:
    """Aggregate summary files into a single DataFrame.
    
    Args:
        file_paths: List of paths to summary files
        summary_type: Type of summary ('alignment', 'merge', or 'dedup')
        
    Returns:
        DataFrame with aggregated summaries (one row per well)
    """
    print(f"Aggregating {len(file_paths)} {summary_type} summary files...")
    
    all_wells = get_all_expected_wells(file_paths)
    print(f"Expected wells: {len(all_wells)}")
    
    aggregated_rows = []
    processed_wells = set()
    
    # Process existing files
    for file_path in file_paths:
        try:
            plate, well = extract_well_id_from_path(file_path)
            well_key = (plate, well)
            
            if summary_type == 'alignment':
                df = load_alignment_summary(file_path)
            else:
                df = load_key_value_summary(file_path, summary_type)
            
            if df is not None and not df.empty:
                # Ensure plate/well columns are present and correct
                df['plate'] = plate
                df['well'] = well
                aggregated_rows.append(df)
                processed_wells.add(well_key)
                print(f"✅ Processed {summary_type} summary for {plate}-{well}")
            else:
                print(f"⚠️  Empty or invalid {summary_type} summary for {plate}-{well}")
                # Create placeholder for invalid file
                placeholder = create_failed_well_placeholder(plate, well, summary_type)
                aggregated_rows.append(placeholder)
                processed_wells.add(well_key)
        
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Create placeholders for missing wells
    missing_wells = set(all_wells) - processed_wells
    for plate, well in missing_wells:
        print(f"⚠️  Creating placeholder for missing {summary_type} summary: {plate}-{well}")
        placeholder = create_failed_well_placeholder(plate, well, summary_type)
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

# Aggregate each summary type
try:
    # Process alignment summaries (already in row format)
    alignment_df = aggregate_summaries(alignment_paths, 'alignment')
    
    # Process merge summaries (convert from key-value to row format)
    merge_df = aggregate_summaries(merge_paths, 'merge')
    
    # Process deduplication summaries (convert from key-value to row format)  
    dedup_df = aggregate_summaries(dedup_paths, 'dedup')
    
    # Save aggregated summaries
    print("\nSaving aggregated summaries...")
    
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
        successful_alignment = len(alignment_df[alignment_df['status'] != 'failed'])
        successful_wells.append(f"Alignment: {successful_alignment}/{len(alignment_df)}")
    
    if not merge_df.empty:
        successful_merge = len(merge_df[merge_df['status'] != 'failed'])  
        successful_wells.append(f"Merge: {successful_merge}/{len(merge_df)}")
        
    if not dedup_df.empty:
        successful_dedup = len(dedup_df[dedup_df['status'] != 'failed'])
        successful_wells.append(f"Dedup: {successful_dedup}/{len(dedup_df)}")
    
    if successful_wells:
        print(f"Success rates: {', '.join(successful_wells)}")
    
except Exception as e:
    print(f"❌ Error during aggregation: {e}")
    raise