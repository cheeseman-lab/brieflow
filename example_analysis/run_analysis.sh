#!/bin/bash

# Set the path to the main Snakefile and the config file
SNAKEFILE="../workflow/Snakefile"
CONFIG_FILE="config/config.yml"

# Run Snakemake with the specified Snakefile and config file
snakemake  --use-conda --snakefile "$SNAKEFILE" --configfile "$CONFIG_FILE" --cores all
