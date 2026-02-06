#!/bin/bash

# Run Brieflow with OME-Zarr exports enabled
snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile "config/config_omezarr.yml" \
    --rerun-triggers mtime \
    --until all_preprocess all_sbs all_phenotype
