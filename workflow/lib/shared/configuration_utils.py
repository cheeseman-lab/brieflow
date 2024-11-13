"""Shared utilties for configuring Brieflow process parameters."""

CONFIG_FILE_HEADER = """
# BrieFlow configuration file

# Defining samples:
#   Samples must be defined in the following TSV files with columns:
#     sbs_samples.tsv: sample_fp, well, tile, cycle
#     phenotype_samples.tsv: sample_fp, well, tile

# Paths:
#   Paths are resolved relative to the directory the workflow is run from

# Suffixes:
#   Each subsection contains a 'suffix' key that defines the folder for saving process files

# Parameters:\n
"""
