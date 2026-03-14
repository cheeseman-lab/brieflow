#!/bin/bash

# Run Brieflow with OME-Zarr exports enabled
snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile "config/config_omezarr.yml" \
    --until all_sbs all_phenotype
        # -R finalize_hcs_sbs finalize_hcs_phenotype
