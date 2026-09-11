"""Adapters between brieflow cluster outputs and the mozzarellm LLM annotation package.

The cluster h5ad written by ``rule format_cluster_anndata`` carries everything
mozzarellm needs -- one row per perturbation, a cluster assignment per Leiden
resolution, and a percentile-rank layer over the features -- so this module
reshapes it into the gene/cluster/feature table mozzarellm consumes, builds the
screen-context JSON from the screen description, and runs the analysis into a
timestamped run directory next to the clustering outputs.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from lib.aggregate.cell_data_utils import control_mask

CLUSTER_GROUP_PREFIX = "cluster_group_"

MOZZARELLM_DIR_NAME = "mozzarellm"

RUN_DIR_PREFIX = "run_"

# run_phate hard-codes its neighbourhood size, so the screen context reports that value
PHATE_KNN = 10

MOZZARELLM_IMPORT_HINT = (
    "mozzarellm is not installed; install it with "
    'python -m pip install "brieflow[mozzarellm]"'
)

PROVIDER_KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "o4": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def cluster_table_from_h5ad(
    h5ad_path,
    leiden_resolution,
    control_key=None,
    n_features=5,
    fdr_threshold=None,
    cluster_ids=None,
):
    """Build the mozzarellm cluster table from a brieflow cluster h5ad.

    The up/down feature lists are the features on which a perturbation sits
    highest and lowest in the ``percentile_rank`` layer, named by
    ``var["feature_name"]`` rather than by index so the model sees real feature
    names. When ``fdr_threshold`` is given and the h5ad carries a bootstrap
    ``fdr`` layer, only features below that threshold are eligible.

    Args:
        h5ad_path (str | Path): Cluster h5ad from ``rule format_cluster_anndata``.
        leiden_resolution (int | float | str): Resolution whose
            ``cluster_group_{res}`` column supplies the cluster assignments.
        control_key (str | list, optional): Control identifier; matching
            perturbations are dropped. Defaults to None (keep everything).
        n_features (int, optional): Features per direction. Defaults to 5.
        fdr_threshold (float, optional): Keep only features with an FDR below
            this value. Defaults to None (no FDR filter).
        cluster_ids (list, optional): Restrict the table to these clusters.
            Defaults to None (every cluster).

    Returns:
        pd.DataFrame: Columns ``gene_symbol``, ``cluster``, ``up_features``,
        ``down_features``, ``phenotypic_strength``.
    """
    adata = ad.read_h5ad(h5ad_path)
    cluster_col = f"{CLUSTER_GROUP_PREFIX}{_resolution_label(leiden_resolution)}"
    if cluster_col not in adata.obs.columns:
        available = [c for c in adata.obs.columns if c.startswith(CLUSTER_GROUP_PREFIX)]
        raise KeyError(f"{h5ad_path} has no {cluster_col}; available: {available}")

    genes = pd.Index(adata.obs_names).astype(str)
    ranks = np.asarray(adata.layers["percentile_rank"], dtype=float)
    feature_names = adata.var["feature_name"].astype(str).to_numpy()

    eligible = np.isfinite(ranks)
    if fdr_threshold is not None and "fdr" in adata.layers:
        fdr = np.asarray(adata.layers["fdr"], dtype=float)
        eligible &= np.isfinite(fdr) & (fdr < fdr_threshold)

    up_features, down_features = [], []
    for row in range(ranks.shape[0]):
        up, down = _extreme_feature_names(
            ranks[row], eligible[row], feature_names, n_features
        )
        up_features.append(up)
        down_features.append(down)

    table = pd.DataFrame(
        {
            "gene_symbol": genes,
            "cluster": adata.obs[cluster_col].to_numpy(),
            "up_features": up_features,
            "down_features": down_features,
            "phenotypic_strength": pd.to_numeric(
                adata.obs.get(
                    "perturbation_auc", pd.Series(np.nan, index=adata.obs_names)
                ),
                errors="coerce",
            ).to_numpy(),
        }
    )

    table = table[table["cluster"].notna()]
    if control_key is not None:
        table = table[~control_mask(table["gene_symbol"], control_key)]
    table["cluster"] = table["cluster"].astype(int)
    if cluster_ids is not None:
        table = table[table["cluster"].isin([int(c) for c in cluster_ids])]

    return table.reset_index(drop=True)


def screen_context_from_screen(screen, config, leiden_resolution):
    """Build the mozzarellm screen-context dict from screen.yaml and config.yml.

    Every key the mozzarellm template marks "required" is filled, falling back
    to a brieflow default whenever the screen description leaves a value null.

    Args:
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        leiden_resolution (int | float | str): Resolution being annotated.

    Returns:
        dict: Screen context ready to be written as JSON.
    """
    experiment = _section(screen, "experiment")
    library = _section(screen, "library")
    collection = _section(screen, "collection")
    cluster_config = _section(config, "cluster")

    channels = _channel_names(screen, config)
    channel_text = ", ".join(channels) if channels else "the phenotype"
    gene_selection = library.get("gene_selection") or "pooled sgRNA library"
    number_of_genes = library.get("number_of_genes")
    if number_of_genes:
        gene_selection = f"{gene_selection} ({number_of_genes} genes)"
    metric = cluster_config.get("phate_distance_metric") or "euclidean"
    resolutions = cluster_config.get("leiden_resolutions") or [leiden_resolution]

    return {
        "assay_type": experiment.get("assay") or "optical pooled screening",
        "target_phenotype": (
            "Morphological cell phenotype — shape, texture and intensity features "
            f"measured across the {channel_text} imaging channels"
        ),
        "organism": experiment.get("organism") or "Homo sapiens",
        "cell_line_or_system": experiment.get("tissue") or "unspecified cell line",
        "perturbation": {
            "type": library.get("vector") or "CRISPR-Cas9 knockout",
            "library_or_reagent": gene_selection,
        },
        "readout": {
            "measurement": "High-content fluorescence imaging of fixed cells",
            "instrument_or_platform": (
                "Optical pooled screening (in situ sequencing + imaging), "
                "processed with brieflow"
            ),
            "primary_metric": (
                "Perturbation-level morphological feature profiles aggregated from "
                "single-cell measurements"
            ),
        },
        "clustering": {
            "method": (
                "Leiden clustering of a PHATE graph over perturbation-level feature "
                f"profiles ({metric} distance)"
            ),
            "parameters": {
                "resolution": str(leiden_resolution),
                "k_neighbors": str(PHATE_KNN),
            },
        },
        "controls": {
            "negative_controls": _control_text(
                library.get("negative_controls"),
                _section(config, "aggregate").get("control_key"),
                "non-targeting sgRNAs",
            ),
            "positive_controls": _control_text(
                library.get("positive_controls"), None, "none designated"
            ),
        },
        "provenance": {
            "dataset_name": (
                collection.get("title")
                or experiment.get("screen_title")
                or "brieflow optical pooled screen"
            ),
            "citation": collection.get("publication_doi") or "unpublished screen",
            "data_source": (
                "brieflow cluster h5ad written by rule format_cluster_anndata"
            ),
        },
        "notes": (
            "Clusters group perturbations with similar morphological phenotypes, so "
            "genes sharing a cluster plausibly perturb the same process. Up and down "
            "features are the features on which a perturbation ranks highest and "
            "lowest across the screen. Clustering was run at resolutions "
            f"{', '.join(str(r) for r in resolutions)}; this context describes "
            f"resolution {leiden_resolution}."
        ),
    }


def organism_id_from_screen(screen):
    """Return the NCBI taxonomy id the screen declares, defaulting to human.

    Args:
        screen (dict): Parsed ``screen.yaml``.

    Returns:
        int: NCBI taxonomy id, e.g. 9606 for ``NCBITaxon:9606``.
    """
    term = _section(screen, "experiment").get("organism_ontology_term_id")
    match = re.search(r"(\d+)", str(term or ""))

    return int(match.group(1)) if match else 9606


def mozzarellm_run_dir(cluster_dir, stamp=None):
    """Return the run directory a mozzarellm run writes into.

    Args:
        cluster_dir (str | Path): Resolution directory holding the clustering outputs.
        stamp (str, optional): Run stamp. Defaults to None (the current time).

    Returns:
        Path: ``<cluster_dir>/mozzarellm/run_<stamp>``.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    return Path(cluster_dir) / MOZZARELLM_DIR_NAME / f"{RUN_DIR_PREFIX}{stamp}"


def latest_mozzarellm_run(cluster_dir):
    """Return the newest mozzarellm run directory holding a cluster JSON.

    Args:
        cluster_dir (str | Path): Resolution directory holding the clustering outputs.

    Returns:
        Path | None: Newest ``run_*`` directory with a ``*_clusters.json``, or None.
    """
    runs = [
        run
        for run in sorted(
            Path(cluster_dir).glob(f"{MOZZARELLM_DIR_NAME}/{RUN_DIR_PREFIX}*")
        )
        if run.is_dir() and any(run.glob("*_clusters.json"))
    ]

    return runs[-1] if runs else None


def run_mozzarellm(
    h5ad_path,
    cluster_dir,
    screen,
    config,
    leiden_resolution,
    model,
    mode="cot",
    mcp=False,
    include_features=True,
    n_features=5,
    fdr_threshold=None,
    temperature=None,
    max_tokens=16000,
    screen_name=None,
    cluster_ids=None,
):
    """Annotate a resolution's clusters with mozzarellm and write the run outputs.

    Args:
        h5ad_path (str | Path): Cluster h5ad to annotate.
        cluster_dir (str | Path): Resolution directory the run is written under.
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        leiden_resolution (int | float | str): Resolution to annotate.
        model (str): Model identifier, e.g. ``claude-sonnet-5``.
        mode (str, optional): Prompt mode. Defaults to "cot".
        mcp (bool, optional): Attach mozzarellm's literature tools. Defaults to False.
        include_features (bool, optional): Feed the up/down feature lists to the
            model. Defaults to True.
        n_features (int, optional): Features per direction. Defaults to 5.
        fdr_threshold (float, optional): FDR cutoff for eligible features.
            Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to None
            (the mozzarellm client default).
        max_tokens (int, optional): Response token budget. Defaults to 16000.
        screen_name (str, optional): Label prefixing the output files. Defaults
            to None (the screen title, else the cluster directory's cell class
            and channel combo).
        cluster_ids (list, optional): Restrict the run to these clusters.
            Defaults to None (every cluster).

    Returns:
        dict: The ``analyze_screen`` result, with ``run_dir`` set to the run directory.

    Raises:
        ValueError: If ``mcp`` and ``include_features`` are both set, which
            mozzarellm does not support.
    """
    if mcp and include_features:
        raise ValueError(
            "mozzarellm supports include_features only for mode='cot' without MCP; "
            "set mcp=False or include_features=False"
        )

    try:
        from mozzarellm.clients.llm_api_clients import create_client
        from mozzarellm.pipeline.screen_analysis import (
            analyze_screen,
            prepare_screen_bundles,
        )
    except ImportError as e:
        raise ImportError(MOZZARELLM_IMPORT_HINT) from e

    screen_name = screen_name or _screen_name(screen, cluster_dir)
    output_dir = Path(cluster_dir) / MOZZARELLM_DIR_NAME
    run_dir = mozzarellm_run_dir(cluster_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cluster_table = cluster_table_from_h5ad(
        h5ad_path,
        leiden_resolution,
        control_key=_section(config, "aggregate").get("control_key"),
        n_features=n_features,
        fdr_threshold=fdr_threshold,
        cluster_ids=cluster_ids,
    )

    screen_context_path = run_dir / "screen_context.json"
    screen_context_path.write_text(
        json.dumps(
            screen_context_from_screen(screen, config, leiden_resolution), indent=2
        ),
        encoding="utf-8",
    )

    bundles = prepare_screen_bundles(
        screen_name=screen_name,
        cluster_table=cluster_table,
        output_dir=output_dir,
        organism_id=organism_id_from_screen(screen),
        feature_columns=["up_features", "down_features"] if include_features else None,
    )
    if cluster_ids is not None:
        wanted = {str(c) for c in cluster_ids}
        bundles = {k: v for k, v in bundles.items() if str(k) in wanted}

    client_kwargs = {"model": model, "max_tokens": max_tokens}
    if temperature is not None:
        client_kwargs["temperature"] = temperature
    api_key = _api_key_for_model(model)
    if api_key:
        client_kwargs["api_key"] = api_key

    result = analyze_screen(
        screen_name=screen_name,
        cluster_to_bundle_map=bundles,
        client=create_client(**client_kwargs),
        run_dir=run_dir,
        screen_context_path=screen_context_path,
        mode=mode,
        mcp=mcp,
        include_features=include_features,
    )
    result["run_dir"] = run_dir

    return result


def _resolution_label(leiden_resolution):
    """Render a resolution the way ``format_cluster_anndata`` names its obs column."""
    value = float(leiden_resolution)

    return str(int(value)) if value.is_integer() else str(value)


def _extreme_feature_names(rank_row, eligible_row, feature_names, n_features):
    """Return the comma-joined highest- and lowest-ranked eligible feature names."""
    indices = np.flatnonzero(eligible_row)
    if indices.size == 0:
        return "", ""

    order = indices[np.argsort(rank_row[indices], kind="stable")]
    # a narrow eligible set is split down the middle so up and down never overlap
    take = (
        min(n_features, order.size // 2) if order.size < 2 * n_features else n_features
    )
    if take == 0:
        return "", ""

    up = ",".join(feature_names[i] for i in order[: -take - 1 : -1])
    down = ",".join(feature_names[i] for i in order[:take])

    return up, down


def _section(mapping, key):
    """Return a nested mapping section, treating a missing or null section as empty."""
    return (mapping or {}).get(key) or {}


def _channel_names(screen, config):
    """Return the phenotype channel names from config, then screen.yaml."""
    channels = _section(config, "phenotype").get("channel_names")
    if channels:
        return [str(c) for c in channels]

    phenotype = _section(screen, "phenotype")
    if phenotype.get("channel_names"):
        return [str(c) for c in phenotype["channel_names"]]

    return [
        str(channel["biological_name"])
        for channel in phenotype.get("channels") or []
        if isinstance(channel, dict) and channel.get("biological_name")
    ]


def _control_text(values, fallback_key, default):
    """Render a control list from screen.yaml, falling back to the config control key."""
    if values:
        return ", ".join(str(v) for v in values)
    if fallback_key:
        keys = (
            fallback_key if isinstance(fallback_key, (list, tuple)) else [fallback_key]
        )
        return ", ".join(str(k) for k in keys)

    return default


def _screen_name(screen, cluster_dir):
    """Return a file-safe screen label, falling back to the cluster directory's identity."""
    name = _section(screen, "experiment").get("screen_title") or _cluster_dir_label(
        cluster_dir
    )

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("_") or "brieflow_screen"


def _cluster_dir_label(cluster_dir):
    """Return ``<cell_class>_<channel_combo>`` for a resolution directory.

    The tree is ``cluster/<combo>/[<compartment>/]<cell_class>/<res>``, so the
    channel combo is the segment below the cluster root rather than a fixed
    offset from the end.
    """
    parts = Path(cluster_dir).resolve().parts
    if len(parts) < 3:
        return "brieflow_screen"

    roots = [i for i, part in enumerate(parts[:-2]) if part == "cluster"]

    return f"{parts[-2]}_{parts[roots[-1] + 1] if roots else parts[-3]}"


def _api_key_for_model(model):
    """Return the API key for the model's provider, or None to let the client look it up."""
    model_lower = str(model).lower()
    for prefix, env_var in PROVIDER_KEY_ENV.items():
        if model_lower.startswith(prefix):
            return os.getenv(env_var)

    return None
