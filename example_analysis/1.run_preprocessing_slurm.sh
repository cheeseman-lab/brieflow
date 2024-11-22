#!/bin/bash

# Set the path to the main Snakefile and the config file
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"
PROFILE_FILE_PATH="slurm/"

# Run only the preprocess rules with cluster configuration
snakemake --use-conda --executor slurm \
    --workflow-profile "$PROFILE_FILE_PATH" \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH" \
    --until all_preprocess