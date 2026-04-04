import pandas as pd

from lib.cluster.format_cluster_anndata import format_cluster_anndata

# Load inputs
features_genes = pd.read_csv(snakemake.input.features_genes, sep="\t")
clustering = pd.read_csv(snakemake.input.clustering, sep="\t")

# Optional bootstrap results
bootstrap_results = None
bootstrap_path = snakemake.input.get("bootstrap_results", None)
if bootstrap_path:
    bootstrap_results = pd.read_csv(bootstrap_path, sep="\t")

# Build AnnData
adata = format_cluster_anndata(
    features_genes=features_genes,
    clustering=clustering,
    perturbation_col=snakemake.params.perturbation_name_col,
    channel_names=snakemake.params.channel_names,
    cell_class=snakemake.wildcards.cell_class,
    channel_combo=snakemake.wildcards.channel_combo,
    leiden_resolution=snakemake.wildcards.leiden_resolution,
    bootstrap_results=bootstrap_results,
)

print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
