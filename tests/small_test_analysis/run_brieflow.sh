#!/bin/bash

# Run the small test analysis.
# Writes OME-Zarr output to brieflow_output/. Pass --tiff to write TIFF output
# instead (same config, all.image_format overridden on the command line) and
# --output-dir DIR to write somewhere else, e.g. to keep a TIFF and a zarr run
# side by side. --zarr is accepted and selects the default.
set -e

FORMAT=""
OUTPUT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tiff) FORMAT="tiff" ;;
        --zarr) FORMAT="" ;;
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}" ;;
        *) echo "Unknown argument: $1 (supported: --tiff, --zarr, --output-dir DIR)" >&2; exit 1 ;;
    esac
    shift
done

OVERRIDES=()
[ -n "$FORMAT" ] && OVERRIDES+=("image_format: $FORMAT")
[ -n "$OUTPUT_DIR" ] && OVERRIDES+=("root_fp: ${OUTPUT_DIR%/}/")
CONFIG_ARGS=()
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    CONFIG_ARGS=(--config "all={$(IFS=,; echo "${OVERRIDES[*]}")}")
fi

snakemake --use-conda --cores all \
    --snakefile "../../workflow/Snakefile" \
    --configfile config/config.yml \
    "${CONFIG_ARGS[@]}" \
    --until all_preprocess all_sbs all_phenotype all_merge all_aggregate all_cluster
