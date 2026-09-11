import pandas as pd

from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal
from lib.shared.image_io import read_image

# Load nuclei data
nuclei_data = read_image(snakemake.input[0])

# Build wildcards dict, synthesizing 'well' from 'row'+'col' in zarr mode
wc = dict(snakemake.wildcards)
if "row" in wc and "col" in wc and "well" not in wc:
    wc["well"] = wc["row"] + wc["col"]

# Extract minimal phenotype information
phenotype_minimal = extract_phenotype_minimal(
    phenotype_data=nuclei_data,
    nuclei_data=nuclei_data,
    wildcards=wc,
)

# Add alignment metrics columns if provided (e.g., phenotype has them, SBS does not)
if len(snakemake.input) > 1:
    alignment_metrics = pd.read_csv(snakemake.input[1], sep="\t")
    # Excludes plate/well/tile as those are already in phenotype_minimal
    metrics_cols = [
        c for c in alignment_metrics.columns if c not in ["plate", "well", "tile"]
    ]
    for col in metrics_cols:
        phenotype_minimal[col] = alignment_metrics[col].iloc[0]

# Attach the per-cell nuclei count when provided (SBS), defaulting to 1 where a cell has no entry
if len(snakemake.input) > 2:
    nuclei_per_cell = pd.read_csv(snakemake.input[2], sep="\t").set_index("cell")
    phenotype_minimal["num_nuclei"] = (
        phenotype_minimal["cell"]
        .map(nuclei_per_cell["num_nuclei"])
        .fillna(1)
        .astype(int)
    )

# save minimal phenotype data
phenotype_minimal.to_csv(snakemake.output[0], index=False, sep="\t")
