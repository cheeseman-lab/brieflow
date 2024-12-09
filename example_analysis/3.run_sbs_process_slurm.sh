#!/bin/bash
#SBATCH --job-name=sbs_process   # Job name
#SBATCH --partition=20                   # Partition name
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=1               # Single CPU for the controller job
#SBATCH --mem=4G                        # Memory for the controller job
#SBATCH --time=72:00:00                 # Time limit (hrs:min:sec)
#SBATCH --output=slurm_output/main/sbs_process-%j.out  # Standard output log

# Set the path to the main Snakefile, the config file, and the workflow profile
BRIEFLOW_PATH="../"
CONFIG_FILE_PATH="config/config.yml"
PROFILE_FILE_PATH="slurm/"

# Create slurm output directories
mkdir -p slurm_output/rule

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

# Run only the preprocess rules
snakemake --use-conda --executor slurm \
    --workflow-profile "$PROFILE_FILE_PATH" \
    --snakefile "${BRIEFLOW_PATH}workflow/Snakefile" \
    --configfile "$CONFIG_FILE_PATH"
    