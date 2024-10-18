import sys
import pickle
import pandas as pd
from workflow.lib.preprocess.preprocess import _extract_metadata_tile

# Load file paths from snakemake input
input_files = snakemake.input  # This will be the list of ND2 file paths from the Snakemake rule

# Extract metadata using the _extract_metadata_tile function
metadata_df = _extract_metadata_tile(input_files)

# Save the extracted metadata to the output path
output_path = snakemake.output[0]  # The output file path from the Snakemake rule

# Save metadata as a pickle file
with open(output_path, 'wb') as f:
    pickle.dump(metadata_df, f)
