#!/bin/bash

# Run the merge process rules
snakemake --use-conda --cores all \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --until all_cluster_process -n
