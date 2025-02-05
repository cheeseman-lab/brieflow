from pathlib import Path

import pandas as pd

from lib.shared.file_utils import parse_filename
from lib.cluster.cluster_eval import aggregate_global_metrics

# aggregate global metrics
aggregated_global_metrics = aggregate_global_metrics(snakemake.input)

# save aggregated global metrics
aggregated_global_metrics.to_csv(snakemake.output[0], sep="\t", index=False)
