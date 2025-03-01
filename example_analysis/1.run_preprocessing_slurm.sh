#!/bin/bash

# Start timing
start_time=$(date +%s)

# Run Snakemake for preprocess rules
snakemake --executor slurm --use-conda \
    --workflow-profile "slurm/" \
    --snakefile "../workflow/Snakefile" \
    --configfile "config/config.yml" \
    --latency-wait 60 \
    --rerun-triggers mtime \
    --until all_preprocess

# End timing and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s" >> slurm/slurm_output/main/preprocessing-$SLURM_JOB_ID.out
