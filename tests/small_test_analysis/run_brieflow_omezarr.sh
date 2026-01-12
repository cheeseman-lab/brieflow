#!/bin/bash

# Fix OpenMP library conflict issue with Cellpose/PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the entire workflow from end to end
# Must specify target rules since there's no default 'rule all'
snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rerun-triggers mtime \
    --until all_preprocess all_sbs all_phenotype all_merge all_aggregate all_cluster
