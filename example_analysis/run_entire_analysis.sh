#!/bin/bash

# Generate a rulegraph of the Snakefile
# NOTE: Uncomment when needed, takes extra computation
snakemake \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --rulegraph | dot -Gdpi=100 -Tpng -o "../images/brieflow_rulegraph.png"

# Run Snakemake with the specified Snakefile and config file
# snakemake --use-conda --executor slurm \
#     --snakefile "../workflow/Snakefile" \
#     --configfile "config/config.yml" \
#     --latency-wait 60 \
#     --rerun-triggers mtime \
#     --until all -n