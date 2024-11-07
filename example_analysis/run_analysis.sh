#!/bin/bash

# Set the path to the main Snakefile and the config file
SNAKEFILE="../workflow/Snakefile"
CONFIG_FILE="config/config.yml"
RULEGRAPH_FILE="../images/brieflow_rulegraph.png"

# Generate a rulegraph of the Snakefile
# snakemake --snakefile "$SNAKEFILE" --configfile "$CONFIG_FILE" --rulegraph | dot -Gdpi=100 -Tpng -o "$RULEGRAPH_FILE"

# Run Snakemake with the specified Snakefile and config file
snakemake  --use-conda --cores all --snakefile "$SNAKEFILE" --configfile "$CONFIG_FILE"
