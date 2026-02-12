import pandas as pd
from tifffile import imread

from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal

# load nuclei data
nuclei_data = imread(snakemake.input[0])

# load alignment metrics
alignment_metrics = pd.read_csv(snakemake.input[1], sep="\t")

# extract minimal phenotype information
phenotype_minimal = extract_phenotype_minimal(
    phenotype_data=nuclei_data,
    nuclei_data=nuclei_data,
    wildcards=snakemake.wildcards,
)

# Add all alignment metrics columns (same value for all cells in tile)
# Excludes plate/well/tile as those are already in phenotype_minimal
metrics_cols = [
    c for c in alignment_metrics.columns if c not in ["plate", "well", "tile"]
]
for col in metrics_cols:
    phenotype_minimal[col] = alignment_metrics[col].iloc[0]

# save minimal phenotype data with alignment metrics
phenotype_minimal.to_csv(snakemake.output[0], index=False, sep="\t")
