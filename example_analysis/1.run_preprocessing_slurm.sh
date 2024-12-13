#!/bin/bash

#SBATCH --job-name=preprocessing   # Job name
#SBATCH --partition=20                   # Partition name
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=1               # Single CPU for the controller job
#SBATCH --mem=4G                        # Memory for the controller job
#SBATCH --time=72:00:00                 # Time limit (hrs:min:sec)
#SBATCH --output=slurm_output/main/preprocessing-%j.out  # Standard output log

# Create slurm output directories
mkdir -p slurm_output/rule

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

# Run Snakemake for preprocess rules
snakemake --executor slurm --use-conda \
    --workflow-profile "slurm/" \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --latency-wait 60 \
    --until all_preprocess
