#!/bin/bash

# Run only the preprocess rules
snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rerun-triggers mtime \
    --until all_preprocess all_sbs all_phenotype all_merge # all_aggregate all_cluster
