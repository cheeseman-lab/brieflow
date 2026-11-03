#!/bin/bash

# Run the small test analysis.
# Defaults to TIFF output. Pass --zarr to write OME-Zarr output instead;
# config_zarr.yml is deep-merged on top of config.yml so both modes run the
# identical pipeline parameters.
set -e

CONFIGFILES="config/config.yml"
for arg in "$@"; do
    case "$arg" in
        --zarr) CONFIGFILES="config/config.yml config/config_zarr.yml" ;;
        *) echo "Unknown argument: $arg (supported: --zarr)" >&2; exit 1 ;;
    esac
done

snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile $CONFIGFILES \
    --until all_preprocess all_sbs all_phenotype all_merge all_aggregate all_cluster
