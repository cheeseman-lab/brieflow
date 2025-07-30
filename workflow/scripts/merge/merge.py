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
    # Add 'tile' and 'site' columns if they don't exist (for compatibility with format_merge.py)
    if 'tile' not in well_merge_data.columns:
        # For enhanced stitching, we don't have individual tiles anymore
        # Set a placeholder value or derive from cell_0/cell_1 if needed
        well_merge_data['tile'] = 1  # Placeholder since we're working with stitched wells
        
    if 'site' not in well_merge_data.columns:
        well_merge_data['site'] = 1  # Placeholder since we're working with stitched wells

    print(f"Enhanced merge data columns: {list(well_merge_data.columns)}")
    print(f"Sample data:")
    print(well_merge_data.head())
    
    if len(well_merge_data) > 0:
        print(f"Distance statistics:")
        print(f"  Mean: {well_merge_data['distance'].mean():.3f}")
        print(f"  Std:  {well_merge_data['distance'].std():.3f}")
        print(f"  Min:  {well_merge_data['distance'].min():.3f}")
        print(f"  Max:  {well_merge_data['distance'].max():.3f}")

else:
    print("No merged cells found")

# Save the merge data (it's already in the final format from enhanced well merge)
well_merge_data.to_parquet(snakemake.output[0])

print(f"Saved enhanced merge data to: {snakemake.output[0]}")
print(f"Total cells saved: {len(well_merge_data)}")