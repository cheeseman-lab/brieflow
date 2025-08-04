import pandas as pd
import numpy as np

from lib.shared.file_utils import validate_dtypes

# Load enhanced well merge data (this is already the merged result)
well_merge_data = validate_dtypes(pd.read_parquet(snakemake.input[0]))

print(f"Loaded enhanced well merge data: {len(well_merge_data)} cells")

# The enhanced well merge already produces the final merged cells
# So we just need to ensure the format is consistent with downstream processing

# Check if we have the expected columns
expected_columns = [
    "plate", "well", "cell_0", "i_0", "j_0", "area_0",
    "cell_1", "i_1", "j_1", "area_1", "distance"
]

missing_columns = [col for col in expected_columns if col not in well_merge_data.columns]
if missing_columns:
    print(f"Warning: Missing columns in well merge data: {missing_columns}")

# If we need to add compatibility columns for downstream processing
if len(well_merge_data) > 0:
    
    # Critical: Ensure tile and site columns exist for format_merge.py compatibility
    if 'tile' not in well_merge_data.columns:
        # Check if we have tile information from the position DataFrames
        if 'tile_0' in well_merge_data.columns:
            well_merge_data['tile'] = well_merge_data['tile_0']  # Use phenotype tile
            print("Using phenotype tile information (tile_0 -> tile)")
        else:
            # Last resort: set to 1 (will work but loses tile specificity)
            well_merge_data['tile'] = 1
            print("Warning: No tile information found, using default tile=1")
        
    if 'site' not in well_merge_data.columns:
        # Site is critical for SBS joining in format_merge.py
        if 'site_1' in well_merge_data.columns:
            well_merge_data['site'] = well_merge_data['site_1']  # Use SBS site
            print("Using SBS site information (site_1 -> site)")
        elif 'tile_1' in well_merge_data.columns:
            well_merge_data['site'] = well_merge_data['tile_1']  # Use SBS tile as site
            print("Using SBS tile as site information (tile_1 -> site)")
        else:
            # Last resort: set to 1 
            well_merge_data['site'] = 1
            print("Warning: No site information found, using default site=1")

    print(f"Enhanced merge data columns: {list(well_merge_data.columns)}")
    
    # Verify we have the critical columns for downstream processing
    critical_cols = ['plate', 'well', 'cell_0', 'cell_1', 'tile', 'site']
    missing_critical = [col for col in critical_cols if col not in well_merge_data.columns]
    
    if missing_critical:
        print(f"ERROR: Missing critical columns for downstream processing: {missing_critical}")
    else:
        print("✅ All critical columns present for downstream processing")
    
    print(f"Sample data:")
    print(well_merge_data[['plate', 'well', 'cell_0', 'cell_1', 'tile', 'site', 'distance']].head())
    
    if len(well_merge_data) > 0:
        print(f"Distance statistics:")
        print(f"  Mean: {well_merge_data['distance'].mean():.3f}")
        print(f"  Std:  {well_merge_data['distance'].std():.3f}")
        print(f"  Min:  {well_merge_data['distance'].min():.3f}")
        print(f"  Max:  {well_merge_data['distance'].max():.3f}")
        
        # Show tile/site distribution
        print(f"Tile distribution: {well_merge_data['tile'].value_counts().to_dict()}")
        print(f"Site distribution: {well_merge_data['site'].value_counts().to_dict()}")

else:
    print("No merged cells found")

# Save the merge data (it's already in the final format from enhanced well merge)
well_merge_data.to_parquet(snakemake.output[0])

print(f"Saved enhanced merge data to: {snakemake.output[0]}")
print(f"Total cells saved: {len(well_merge_data)}")