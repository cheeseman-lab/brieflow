#!/bin/bash
#SBATCH --job-name=preprocessing   # Job name
#SBATCH --partition=20                   # Partition name
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=1               # Single CPU for the controller job
#SBATCH --mem=4G                        # Memory for the controller job
#SBATCH --time=72:00:00                 # Time limit (hrs:min:sec)
#SBATCH --output=slurm_output/main/preprocessing-%j.out  # Standard output log

# Set the path to the main Snakefile, the config file, and the workflow profile
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"
PROFILE_FILE_PATH="slurm/"

# Create slurm output directories
mkdir -p slurm_output/rule

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

# Run Snakemake
snakemake --executor slurm --use-conda \
    --workflow-profile "$PROFILE_FILE_PATH" \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH" \
    --latency-wait 60 \
    --until all_preprocess
    