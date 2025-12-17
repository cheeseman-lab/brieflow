"""Combine mozzarellm results from individual cluster analyses.

This script merges the JSON outputs from parallel cluster analyses
into a single combined result file.
"""

import json
from pathlib import Path

# Get all cluster result files
cluster_result_files = snakemake.input.cluster_results

print(f"Combining {len(cluster_result_files)} cluster results")

# Combine all cluster results
combined_results = {
    "clusters": {},
    "metadata": {
        "model_name": snakemake.params.model_name,
        "num_clusters": len(cluster_result_files),
    },
}

for result_file in cluster_result_files:
    result_path = Path(result_file)

    # Extract cluster ID from filename (format: cluster_X.json)
    cluster_id = result_path.stem.replace("cluster_", "")

    try:
        with open(result_file, "r") as f:
            cluster_result = json.load(f)

        # Add to combined results
        combined_results["clusters"][cluster_id] = cluster_result

        print(f"Added cluster {cluster_id} results")

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load results for cluster {cluster_id}: {e}")
        combined_results["clusters"][cluster_id] = {"error": str(e)}

# Save combined results
with open(snakemake.output[0], "w") as f:
    json.dump(combined_results, f, indent=2)

print(f"Combined results saved to {snakemake.output[0]}")
