#!/bin/bash
#SBATCH --job-name=all    # Job name
#SBATCH --partition=20                  # Partition name
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=1              # Single CPU for the controller job
#SBATCH --mem=10G                       # Memory for the controller job
#SBATCH --time=72:00:00                # Time limit (hrs:min:sec)
#SBATCH --output=slurm/slurm_output/main/all-%j.out  # Standard output log

# Start timing
start_time=$(date +%s)

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
    --configfile "config/config.yml" \
    --latency-wait 60

# End timing and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s" >> slurm/slurm_output/main/all-$SLURM_JOB_ID.out
