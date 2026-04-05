import pandas as pd

from lib.cluster.format_cluster_anndata import format_cluster_anndata

# Load inputs
features_genes = pd.read_csv(snakemake.input.features_genes, sep="\t")

# Load clustering files for all resolutions
clustering_paths = snakemake.input.clustering
if isinstance(clustering_paths, str):
    clustering_paths = [clustering_paths]
leiden_resolutions = snakemake.params.leiden_resolutions

clusterings = {}
for path, res in zip(clustering_paths, leiden_resolutions):
    df = pd.read_csv(path, sep="\t")
    print(f"Resolution {res}: {df.shape}")
    clusterings[str(res)] = df

# Bootstrap results
bootstrap_results = None
try:
    bootstrap_path = snakemake.input.bootstrap_results
    if bootstrap_path:
        bootstrap_results = pd.read_csv(bootstrap_path, sep="\t")
        print(f"Bootstrap results: {bootstrap_results.shape}")
except AttributeError:
    pass

# Build AnnData
adata = format_cluster_anndata(
    features_genes=features_genes,
    clusterings=clusterings,
    perturbation_col=snakemake.params.perturbation_name_col,
    channel_names=snakemake.params.channel_names,
    cell_class=snakemake.wildcards.cell_class,
    channel_combo=snakemake.wildcards.channel_combo,
    bootstrap_results=bootstrap_results,
)

print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
