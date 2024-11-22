#!/bin/bash

# Set the path to the main Snakefile and the config file
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"
PROFILE_FILE_PATH="config/"

# Run only the preprocess rules with cluster configuration
snakemake --use-conda --cores all \
    --profile "$PROFILE_FILE_PATH" \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH" \
    --until all_preprocess