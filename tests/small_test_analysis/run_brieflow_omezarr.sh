#!/bin/bash

# Fix OpenMP library conflict issue with Cellpose/PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE

# Run only the rules responsible for writing OME-Zarr outputs
snakemake --use-conda --cores all \
    --snakefile "/Users/cspeters/brieflow/workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rerun-triggers mtime \
    --until convert_sbs_omezarr \
    --until convert_phenotype_omezarr
