#!/bin/bash
#SBATCH --job-name=test_aggregate_slurm
#SBATCH --output=test_aggregate_slurm-%j.out
#SBATCH --error=test_aggregate_slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G

# Create slurm output directories
mkdir -p slurm_output/rule

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

sh 8.run_aggregate_process.sh
