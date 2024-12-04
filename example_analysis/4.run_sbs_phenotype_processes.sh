#!/bin/bash

# Set the path to the main Snakefile and the config file
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"

# TODO: Run both processes when done with testing

# Run the SBS/phenotype process rules
snakemake --use-conda --cores all \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH" \
    --until all_sbs_process #all_phenotype_process
