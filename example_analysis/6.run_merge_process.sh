#!/bin/bash

# TODO: update below to run the entire merge process
# Run the merge process rules
snakemake --use-conda --cores all \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --force fast_alignment