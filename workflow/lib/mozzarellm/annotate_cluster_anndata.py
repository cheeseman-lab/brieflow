"""Merge a mozzarellm run's calls back into the clustering's h5ad."""

import json
from pathlib import Path

import anndata as ad
import pandas as pd

from lib.mozzarellm.annotate_clusters import cluster_group_column


# obs column each mozzarellm per-gene call lands in
GENE_ANNOTATION_COLUMNS = {
    "mozzarellm_category": "category",
    "mozzarellm_subclass": "subclass",
    "mozzarellm_rationale": "rationale",
}

CLUSTER_ANNOTATION_KEYS = ("dominant_process", "pathway_confidence", "summary")


def annotate_cluster_anndata(
    h5ad_path,
    clusters_json_path,
    genes_csv_path,
    leiden_resolution,
    run_name=None,
    screen_name=None,
):
    """Merge a mozzarellm run's calls into the clustering's h5ad.

    The per-gene calls land in ``obs`` as ``mozzarellm_category``,
    ``mozzarellm_subclass`` and ``mozzarellm_rationale``, blank for
    perturbations the run left unclassified. The cluster-level calls land in
    ``uns["mozzarellm"]["clusters"]``, keyed by the cluster label held in the
    ``uns["mozzarellm"]["cluster_column"]`` obs column, so a reader can go from
    a perturbation to its cluster's call without the run directory.

    Args:
        h5ad_path (str | Path): Cluster h5ad from ``rule format_cluster_anndata``.
        clusters_json_path (str | Path): The run's ``<screen>_clusters.json``.
        genes_csv_path (str | Path): The run's ``<screen>_genes.csv``.
        leiden_resolution (int | float | str): Resolution the run annotated.
        run_name (str, optional): Run directory name, recorded in ``uns``.
            Defaults to None.
        screen_name (str, optional): Screen label, recorded in ``uns``. Defaults
            to None.

    Returns:
        ad.AnnData: The annotated object, ready to write.
    """
    adata = ad.read_h5ad(h5ad_path)
    cluster_col = cluster_group_column(adata, leiden_resolution, h5ad_path)

    genes = pd.read_csv(genes_csv_path)
    calls = (
        genes.drop_duplicates(subset="gene").set_index("gene")
        if "gene" in genes.columns
        else pd.DataFrame()
    )
    obs_genes = pd.Series(pd.Index(adata.obs_names).astype(str), index=adata.obs_names)
    for column, source in GENE_ANNOTATION_COLUMNS.items():
        values = calls[source] if source in calls.columns else pd.Series(dtype="object")
        adata.obs[column] = obs_genes.map(values).fillna("").astype(str)

    clusters = json.loads(Path(clusters_json_path).read_text(encoding="utf-8"))
    clusters = clusters.get("clusters", clusters)
    adata.uns["mozzarellm"] = {
        "run_name": str(run_name or ""),
        "screen_name": str(screen_name or ""),
        "leiden_resolution": str(leiden_resolution),
        "cluster_column": cluster_col,
        "clusters": {
            str(cluster_id): {
                key: str(call.get(key) or "") for key in CLUSTER_ANNOTATION_KEYS
            }
            for cluster_id, call in clusters.items()
        },
    }

    return adata
