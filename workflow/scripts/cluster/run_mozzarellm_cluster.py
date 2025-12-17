"""Run mozzarellm analysis on a single cluster.

This script processes one cluster at a time, enabling parallel execution
across all clusters in a clustering result.
"""

import json

import pandas as pd
from mozzarellm import analyze_gene_clusters, reshape_to_clusters
from mozzarellm.configs import DEFAULT_ANTHROPIC_CONFIG
from mozzarellm.prompts import ROBUST_CLUSTER_PROMPT, ROBUST_SCREEN_CONTEXT

# Load job file to get cluster ID and input file path
with open(snakemake.input.job_file, "r") as f:
    job_data = json.load(f)

cluster_id = job_data["cluster_id"]
input_file = job_data["input_file"]

print(f"Processing cluster {cluster_id} from {input_file}")

# Load cluster data and filter to this cluster
cluster_df = pd.read_csv(input_file, sep="\t")
cluster_subset = cluster_df[cluster_df["cluster"] == cluster_id].copy()

print(f"Cluster {cluster_id} has {len(cluster_subset)} genes")

# Get configuration parameters
gene_col = snakemake.params.perturbation_name_col
model_name = snakemake.params.model_name

# Get prompts from config (if provided) or use defaults
screen_context = snakemake.params.screen_context
cluster_analysis_prompt = snakemake.params.cluster_analysis_prompt

if screen_context is None:
    screen_context = ROBUST_SCREEN_CONTEXT
if cluster_analysis_prompt is None:
    cluster_analysis_prompt = ROBUST_CLUSTER_PROMPT

# Reshape for mozzarellm
# Note: reshape_to_clusters expects a dataframe with gene and cluster columns
llm_cluster_df, llm_uniprot_df = reshape_to_clusters(
    input_df=cluster_subset,
    gene_col=gene_col,
    cluster_col="cluster",
    uniprot_col="uniprot_function",
)

# Output path (remove .json extension for mozzarellm internal naming)
output_base = str(snakemake.output.json_output).replace(".json", "")

# Run analysis on this single cluster
results = analyze_gene_clusters(
    input_df=llm_cluster_df,
    model_name=model_name,
    config_dict=DEFAULT_ANTHROPIC_CONFIG,
    screen_context=screen_context,
    cluster_analysis_prompt=cluster_analysis_prompt,
    gene_annotations_df=llm_uniprot_df,
    batch_size=1,  # Single cluster, so batch_size=1
    output_file=output_base,
    save_outputs=True,
    outputs_to_generate=["json"],
)

print(f"Completed analysis for cluster {cluster_id}")
