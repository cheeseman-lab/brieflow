#!/bin/bash

# TODO: update below to run the entire aggregate process
# Run the merge process rules
snakemake --use-conda --cores all \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --force process_mitotic_gene_data process_interphase_gene_data process_all_gene_data