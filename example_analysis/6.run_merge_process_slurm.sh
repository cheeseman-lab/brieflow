#!/bin/bash
#SBATCH --job-name=merge_process        # Job name
#SBATCH --partition=20                  # Partition name
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=1              # Single CPU for the controller job
#SBATCH --mem=10G                       # Memory for the controller job
#SBATCH --time=72:00:00                # Time limit (hrs:min:sec)
#SBATCH --output=slurm/slurm_output/main/merge_process-%j.out  # Standard output log

# Start timing
start_time=$(date +%s)

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate brieflow_workflows

# Run the merge process rules
snakemake --executor slurm --use-conda \
    --workflow-profile "slurm/" \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --latency-wait 60 \
    --until all_merge_process

# End timing and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s" >> slurm/slurm_output/main/merge_process-$SLURM_JOB_ID.out
