"""Run mozzarellm over one clustering's clusters and write the run outputs.

The cluster h5ad written by ``rule format_cluster_anndata`` carries everything
mozzarellm needs -- one row per perturbation, a cluster assignment per Leiden
resolution, and a percentile-rank layer over the features -- so this module
reshapes it into the gene/cluster/feature table mozzarellm consumes and runs the
analysis into a run directory next to the clustering outputs. The screen context
the run reasons from is written before the stage by
``lib.mozzarellm.selection``; this module only consumes it.
"""

import json
import os
import re
from datetime import datetime
from inspect import signature
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from lib.shared.compartment_utils import get_compartment_combo


CLUSTER_GROUP_PREFIX = "cluster_group_"

MOZZARELLM_DIR_NAME = "mozzarellm"

RUN_DIR_PREFIX = "run_"

DEFAULT_ORGANISM_ID = 9606

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


def run_mozzarellm(
    h5ad_path,
    cluster_dir,
    leiden_resolution,
    model,
    screen_context,
    mode="cot",
    mcp=True,
    annotation_source="affinage_then_uniprot",
    include_features="auto",
    include_strength="auto",
    n_features=5,
    fdr_threshold=None,
    temperature=None,
    max_tokens=64000,
    max_workers=None,
    screen_name=None,
    organism_id=None,
    run_name=None,
    cluster_ids=None,
    resume=False,
    dry_run=False,
):
    """Annotate a resolution's clusters with mozzarellm and write the run outputs.

    Args:
        h5ad_path (str | Path): Cluster h5ad to annotate.
        cluster_dir (str | Path): Resolution directory the run is written under.
        leiden_resolution (int | float | str): Resolution to annotate.
        model (str): Model identifier, e.g. ``claude-sonnet-5``.
        screen_context (dict): Screen context the run reasons from, written by
            the analyze notebook and read back from the combo table's context file.
        mode (str, optional): Prompt mode. Defaults to "cot", mozzarellm's
            benchmark-selected delivery format.
        mcp (bool, optional): Attach mozzarellm's PubMed literature tools, which
            fill in genes whose annotation is blank. Defaults to True, the
            benchmark-selected configuration.
        annotation_source (str, optional): Which functional annotation the
            evidence bundles carry -- "affinage" (Affinage mechanistic
            narratives), "uniprot" (UniProt FUNCTION comments), "both" (each as
            its own column), or "affinage_then_uniprot". Defaults to
            "affinage_then_uniprot": Affinage for every gene, then UniProt for
            the genes whose Affinage annotation is absent, empty, or a refusal
            narrative, so the prompt stays close to pure Affinage while a gene
            UniProt describes well cannot reach the model blank and be called a
            dark gene. Each gene carries an ``annotation_source`` field saying
            which source supplied its text. The three pure sources backfill
            nothing, so a gene one source has nothing for reaches the model as
            a visible gap. Stable accessions always come from UniProt, since
            they are UniProt identifiers.
        include_features (bool | str, optional): Feed the up/down feature lists
            to the model. Defaults to "auto" (on when the bundles carry them).
        include_strength (bool | str, optional): Feed the per-gene perturbation
            strength ranks to the model. Defaults to "auto" (on when the bundles
            carry them).
        n_features (int, optional): Features per direction. Defaults to 5.
        fdr_threshold (float, optional): FDR cutoff for eligible features.
            Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to None
            (the mozzarellm client default).
        max_tokens (int, optional): Response token budget. Defaults to 64000,
            the ceiling mozzarellm's feature-augmented runs need.
        max_workers (int, optional): Clusters to answer concurrently. Defaults
            to None (mozzarellm's own default); ignored by installs whose
            ``analyze_screen`` does not take it yet.
        screen_name (str, optional): Label prefixing the output files. Defaults
            to None (the cluster directory's cell class and channel combo).
        organism_id (int, optional): NCBI taxonomy id for the UniProt lookups.
            Defaults to None (the id the screen context declares).
        run_name (str, optional): Fixed run directory name. Defaults to None
            (a timestamped directory).
        cluster_ids (list, optional): Restrict the run to these clusters.
            Defaults to None (every cluster).
        resume (bool, optional): Reuse clusters already answered in
            ``run_dir/traces``. Defaults to False.
        dry_run (bool, optional): Assemble the prompts and report the estimated
            cost without calling the model. Defaults to False.

    Returns:
        dict: The ``analyze_screen`` result, with ``run_dir`` set to the run directory.
    """
    try:
        from mozzarellm.clients.llm_api_clients import create_client
        from mozzarellm.pipeline.screen_analysis import (
            analyze_screen,
            prepare_screen_bundles,
        )
    except ImportError as e:
        raise ImportError(MOZZARELLM_IMPORT_HINT) from e

    screen_name = screen_name or _screen_name(cluster_dir)
    output_dir = Path(cluster_dir) / MOZZARELLM_DIR_NAME
    if run_name:
        run_dir = mozzarellm_run_dir(cluster_dir, run_name=run_name)
    elif resume:
        run_dir = latest_mozzarellm_run(cluster_dir) or mozzarellm_run_dir(cluster_dir)
    else:
        run_dir = mozzarellm_run_dir(cluster_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cluster_table = cluster_table_from_h5ad(
        h5ad_path,
        leiden_resolution,
        n_features=n_features,
        fdr_threshold=fdr_threshold,
        cluster_ids=cluster_ids,
    )

    if organism_id is None:
        organism_id = organism_id_from_context(screen_context)
    # mozzarellm takes the context in memory; the copy on disk is the run's record
    (run_dir / "screen_context.json").write_text(
        json.dumps(screen_context, indent=2), encoding="utf-8"
    )

    bundles = prepare_screen_bundles(
        screen_name=screen_name,
        cluster_table=cluster_table,
        output_dir=output_dir,
        organism_id=organism_id,
        source=annotation_source,
        feature_columns=(
            None if include_features is False else ["up_features", "down_features"]
        ),
        strength_column=None if include_strength is False else "perturbation_auc",
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

    analyze_kwargs = {}
    # max_workers is newer than the pinned mozzarellm, so older installs just run serially
    if (
        max_workers is not None
        and "max_workers" in signature(analyze_screen).parameters
    ):
        analyze_kwargs["max_workers"] = max_workers

    result = analyze_screen(
        screen_name=screen_name,
        cluster_to_bundle_map=bundles,
        client=create_client(**client_kwargs),
        run_dir=run_dir,
        screen_context=screen_context,
        mode=mode,
        mcp=mcp,
        include_features=include_features,
        include_strength=include_strength,
        resume=resume,
        dry_run=dry_run,
        **analyze_kwargs,
    )
    result["run_dir"] = run_dir

    return result


def cluster_table_from_h5ad(
    h5ad_path,
    leiden_resolution,
    n_features=5,
    fdr_threshold=None,
    cluster_ids=None,
):
    """Build the mozzarellm cluster table from a brieflow cluster h5ad.

    The up/down feature lists are the features on which a perturbation sits
    highest and lowest in the ``percentile_rank`` layer, named by
    ``var["feature_name"]`` rather than by index so the model sees real feature
    names. When ``fdr_threshold`` is given and the h5ad carries a bootstrap
    ``fdr`` layer, only features below that threshold are eligible. Control
    perturbations stay in the table; mozzarellm recognizes them by name.

    Args:
        h5ad_path (str | Path): Cluster h5ad from ``rule format_cluster_anndata``.
        leiden_resolution (int | float | str): Resolution whose
            ``cluster_group_{res}`` column supplies the cluster assignments.
        n_features (int, optional): Features per direction. Defaults to 5.
        fdr_threshold (float, optional): Keep only features with an FDR below
            this value. Defaults to None (no FDR filter).
        cluster_ids (list, optional): Restrict the table to these clusters.
            Defaults to None (every cluster).

    Returns:
        pd.DataFrame: Columns ``gene_symbol``, ``cluster``, ``up_features``,
        ``down_features``, ``perturbation_auc``.
    """
    adata = ad.read_h5ad(h5ad_path)
    cluster_col = cluster_group_column(adata, leiden_resolution, h5ad_path)

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
            "perturbation_auc": pd.to_numeric(
                adata.obs.get(
                    "perturbation_auc", pd.Series(np.nan, index=adata.obs_names)
                ),
                errors="coerce",
            ).to_numpy(),
        }
    )

    table = table[table["cluster"].notna()]
    table["cluster"] = table["cluster"].astype(int)
    if cluster_ids is not None:
        table = table[table["cluster"].isin([int(c) for c in cluster_ids])]

    return table.reset_index(drop=True)


def cluster_group_column(adata, leiden_resolution, h5ad_path=""):
    """Return the obs column holding a resolution's cluster assignments.

    Args:
        adata (ad.AnnData): Cluster h5ad from ``rule format_cluster_anndata``.
        leiden_resolution (int | float | str): Resolution being read.
        h5ad_path (str | Path, optional): Path named in the error. Defaults to "".

    Returns:
        str: The ``cluster_group_{res}`` column name.
    """
    value = float(leiden_resolution)
    label = str(int(value)) if value.is_integer() else str(value)
    column = f"{CLUSTER_GROUP_PREFIX}{label}"
    if column not in adata.obs.columns:
        available = [c for c in adata.obs.columns if c.startswith(CLUSTER_GROUP_PREFIX)]
        raise KeyError(f"{h5ad_path} has no {column}; available: {available}")

    return column


def mozzarellm_run_dir(cluster_dir, stamp=None, run_name=None):
    """Return the run directory a mozzarellm run writes into.

    Args:
        cluster_dir (str | Path): Resolution directory holding the clustering outputs.
        stamp (str, optional): Run stamp. Defaults to None (the current time).
        run_name (str, optional): Fixed run directory name, for callers such as
            the pipeline rule that need a path snakemake can predict. Defaults
            to None (a timestamped ``run_<stamp>``).

    Returns:
        Path: ``<cluster_dir>/mozzarellm/<run_name>``, else ``.../run_<stamp>``.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"{RUN_DIR_PREFIX}{stamp}"

    return Path(cluster_dir) / MOZZARELLM_DIR_NAME / name


def latest_mozzarellm_run(cluster_dir):
    """Return the mozzarellm run directory a reader should display.

    mozzarellm writes ``latest.json`` beside the run directories after a
    successful run; that pointer wins, and the newest ``run_*`` holding a
    cluster JSON is the fallback for runs written before the pointer existed.

    Args:
        cluster_dir (str | Path): Resolution directory holding the clustering outputs.

    Returns:
        Path | None: The pointed-to run directory, else the newest ``run_*``
        with a ``*_clusters.json``, else None.
    """
    pointer = Path(cluster_dir) / MOZZARELLM_DIR_NAME / "latest.json"
    if pointer.exists():
        named = json.loads(pointer.read_text()).get("run_dir")
        run = pointer.parent / str(named) if named else None
        if run is not None and run.is_dir() and any(run.glob("*_clusters.json")):
            return run

    runs = [
        run
        for run in sorted(
            Path(cluster_dir).glob(f"{MOZZARELLM_DIR_NAME}/{RUN_DIR_PREFIX}*")
        )
        if run.is_dir() and any(run.glob("*_clusters.json"))
    ]

    return runs[-1] if runs else None


def organism_id_from_context(screen_context):
    """Return the NCBI taxonomy id a written screen context carries.

    Lets a job that reads a pre-written context resolve the same taxonomy id the
    screen description would have given, without reading screen.yaml itself.

    Args:
        screen_context (dict): Screen context loaded from its JSON.

    Returns:
        int: NCBI taxonomy id, defaulting to human.
    """
    term = (screen_context or {}).get("organism_ontology_term_id")
    match = re.search(r"(\d+)", str(term or ""))

    return int(match.group(1)) if match else DEFAULT_ORGANISM_ID


def mozzarellm_row_value(
    combos, column, wildcards, split_by_compartment, default_compartment_combo
):
    """Return one column of the combo-table row the current job's wildcards select.

    Args:
        combos (pd.DataFrame): The normalized mozzarellm combo table.
        column (str): Column to read, e.g. ``screen_context_fp``.
        wildcards: Snakemake wildcards for the current job.
        split_by_compartment (bool): Whether compartment-specific paths are enabled.
        default_compartment_combo (str): Combo to use when splitting is disabled.

    Returns:
        str: The row's value in ``column``.
    """
    selection = {
        "cell_class": wildcards.cell_class,
        "channel_combo": wildcards.channel_combo,
        "compartment_combo": get_compartment_combo(
            wildcards, split_by_compartment, default_compartment_combo
        ),
        "leiden_resolution": wildcards.leiden_resolution,
    }
    rows = combos
    for key, value in selection.items():
        rows = rows[rows[key].astype(str) == str(value)]
    if rows.empty:
        raise KeyError(f"no mozzarellm combo row for {selection}")

    return str(rows.iloc[0][column])


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


def _api_key_for_model(model):
    """Return the API key for the model's provider, or None to let the client look it up."""
    model_lower = str(model).lower()
    for prefix, env_var in PROVIDER_KEY_ENV.items():
        if model_lower.startswith(prefix):
            return os.getenv(env_var)

    return None


def _screen_name(cluster_dir):
    """Return a file-safe screen label built from the cluster directory's identity."""
    name = _cluster_dir_label(cluster_dir)

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
