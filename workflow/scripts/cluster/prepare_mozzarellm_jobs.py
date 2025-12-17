"""Prepare mozzarellm jobs by creating job files for each cluster.

This is a checkpoint script that reads the clustering output and creates
individual job files for parallel processing of each cluster.
"""

import json
from pathlib import Path

import pandas as pd

# Load clustering results
cluster_df = pd.read_csv(snakemake.input.cluster_file, sep="\t")

# Get unique cluster IDs
cluster_ids = sorted(cluster_df["cluster"].unique().tolist())

print(f"Found {len(cluster_ids)} clusters to process")

# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

# Create a job file for each cluster
for cluster_id in cluster_ids:
    job_file = output_dir / f"cluster_{cluster_id}.json"
    job_data = {
        "cluster_id": int(cluster_id),
        "input_file": str(snakemake.input.cluster_file),
    }
    with open(job_file, "w") as f:
        json.dump(job_data, f)

print(f"Created {len(cluster_ids)} job files in {output_dir}")
