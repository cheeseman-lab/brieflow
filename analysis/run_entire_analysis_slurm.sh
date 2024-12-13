#!/bin/bash

#SBATCH --job-name=all   # Job name
#SBATCH --partition=20                   # Partition name
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=1               # Single CPU for the controller job
#SBATCH --mem=4G                        # Memory for the controller job
#SBATCH --time=72:00:00                 # Time limit (hrs:min:sec)
#SBATCH --output=slurm_output/main/all-%j.out  # Standard output log

# Create slurm output directories
mkdir -p slurm_output/rule

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

# Generate a rulegraph of the Snakefile
# NOTE: Uncomment when needed, takes extra computation
# snakemake \
#     --snakefile "../workflow/Snakefile" \
#     --configfile "config/config.yml" \
#     --rulegraph | dot -Gdpi=100 -Tpng -o "../images/brieflow_rulegraph.png"

# Run Snakemake with the specified Snakefile and config file
snakemake --use-conda --executor slurm \
    --workflow-profile "slurm/" \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml"
