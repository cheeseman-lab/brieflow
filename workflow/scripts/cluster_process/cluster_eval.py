import pandas as pd

# from lib.cluster.cluster_eval import aggregate_resolution_metrics

# load global metrics from each channel combo and dataset combination
channel_combos = []
datasets = []
global_metrics = []
for global_metrics_fp in snakemake.input:
    series = pd.read_csv(global_metrics_fp, sep="\t", index_col=0, header=None)
    global_metrics.append(series)

print(global_metrics)
