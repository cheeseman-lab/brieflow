#!/bin/bash

# Set the path to the main Snakefile and the config file
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"

# Run only the preprocess rules
snakemake --use-conda --cores all \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH" \