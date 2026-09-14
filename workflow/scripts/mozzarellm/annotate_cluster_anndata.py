from lib.cluster.mozzarellm_io import annotate_cluster_anndata

adata = annotate_cluster_anndata(
    snakemake.input.cluster_anndata,
    snakemake.input.clusters_json,
    snakemake.input.genes_csv,
    snakemake.params.leiden_resolution,
    run_name=snakemake.params.run_name,
    screen_name=snakemake.params.screen_name,
)

print(f"\n{adata}")
adata.write_h5ad(snakemake.output[0])
print(f"Saved to {snakemake.output[0]}")
