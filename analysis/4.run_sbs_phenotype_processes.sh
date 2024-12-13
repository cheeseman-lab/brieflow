#!/bin/bash

# Run the SBS/phenotype process rules
snakemake --use-conda --cores all \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --until all_sbs_process all_phenotype_process
