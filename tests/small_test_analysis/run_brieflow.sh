#!/bin/bash

# Run the small test analysis.
# Defaults to OME-Zarr output. Pass --tiff to write TIFF output instead;
# config_tiff.yml is deep-merged on top of config.yml so both modes run the
# identical pipeline parameters. --zarr is accepted and selects the default.
set -e

CONFIGFILES="config/config.yml"
for arg in "$@"; do
    case "$arg" in
        --tiff) CONFIGFILES="config/config.yml config/config_tiff.yml" ;;
        --zarr) CONFIGFILES="config/config.yml" ;;
        *) echo "Unknown argument: $arg (supported: --tiff, --zarr)" >&2; exit 1 ;;
    esac
done

snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile $CONFIGFILES \
    --until all_preprocess all_sbs all_phenotype all_merge all_aggregate all_cluster
