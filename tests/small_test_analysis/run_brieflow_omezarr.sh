#!/bin/bash

# Run only the rules responsible for writing OME-Zarr outputs
snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rerun-triggers mtime \
    --until convert_sbs_omezarr \
    --until convert_phenotype_omezarr
