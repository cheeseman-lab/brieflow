import sys
import pickle
import pandas as pd

# Snakemake automatically passes the input and output file paths via snakemake.input and snakemake.output
input_filepaths = snakemake.input  # This will be the list of file paths passed from the rule
output_filepath = snakemake.output[0]  # This is the output path specified in the Snakemake rule

# Save the list of filepaths to a pickle file
with open(output_filepath, 'wb') as f:
    pickle.dump(input_filepaths, f)
