"""Choose the clusterings mozzarellm annotates and write what the stage reads.

The analyze notebook runs these before the stage: it expands the operator's
selection into one row per clustering, derives a screen context for each row
from ``screen.yaml`` and ``config.yml``, writes those contexts as JSON and
writes ``config/mozzarellm_combo.tsv`` pointing each row at its own context.
Contexts are derived once as a starting point and then belong to the operator,
so an existing file is never overwritten unless a caller asks for it.
"""

import json
from itertools import product
from pathlib import Path

import pandas as pd

from lib.mozzarellm.annotate_clusters import (
    DEFAULT_ORGANISM_ID,
    organism_id_from_context,
)
from lib.shared.file_utils import get_filename


CHANNEL_COMBO_DELIMITER = "_"

DEFAULT_DNA_CHANNEL = "DAPI"

OPTIMAL_RESOLUTION = "optimal"

# the columns that name a clustering, in the order the combo table carries them
SELECTION_COLUMNS = ["cell_class", "channel_combo", "compartment_combo"]

# panel tiers the lab's policy annotates; a DNA-plus-one-marker combo is dropped
PANEL_TIERS_ANNOTATED = (
    "full panel",
    "grouped panel",
    "single marker",
    "DNA-only baseline",
)

# run_phate hard-codes its neighbourhood size, so the screen context reports that value
PHATE_KNN = 10


def expand_mozzarellm_selection(
    cluster_combos,
    selection,
    annotate_selected=True,
    extra_clusterings=(),
    optimal_resolutions=None,
    split_by_compartment=False,
):
    """Expand the notebook's mozzarellm selection into one row per clustering.

    The notebook's current clustering is one row; every entry of
    ``extra_clusterings`` adds more, inheriting each key it omits from that
    selection. Any value may be a list, which expands as a cross product, so
    ``{"channel_combo": "DAPI_WGA", "leiden_resolution": [6, 8]}`` is two rows.
    A ``leiden_resolution`` of ``"optimal"`` is resolved against the
    benchmark-derived picks rather than being an implicit fallback.

    Args:
        cluster_combos (pd.DataFrame): The cluster phase's combo table, which
            names every clustering that exists.
        selection (dict): The notebook's current ``cell_class``,
            ``channel_combo``, ``compartment_combo`` and ``leiden_resolution``.
        annotate_selected (bool, optional): Keep the current selection as a row.
            Defaults to True.
        extra_clusterings (list, optional): Additional rows, each a dict of any
            of the selection keys. Defaults to () (no extra rows).
        optimal_resolutions (dict, optional): ``find_optimal_resolution``
            results keyed ``{cell_class}_{channel_combo}``, which
            ``"optimal"`` resolves through. Defaults to None.
        split_by_compartment (bool, optional): Whether compartment-specific
            paths are enabled. Defaults to False.

    Returns:
        pd.DataFrame: Columns ``cell_class``, ``channel_combo``,
        ``leiden_resolution`` and, when splitting by compartment,
        ``compartment_combo``.
    """
    columns = _selection_columns(split_by_compartment)
    rows = [dict(selection)] if annotate_selected else []
    for entry in extra_clusterings or []:
        rows.extend(_expand_entry(entry, selection, columns))

    available = _available_clusterings(cluster_combos, split_by_compartment)
    expanded = []
    for row in rows:
        resolved = {key: row.get(key) for key in columns}
        if None in resolved.values():
            raise ValueError(f"mozzarellm selection row is incomplete: {resolved}")
        clustering = tuple(
            str(resolved[key]) for key in columns if key != "leiden_resolution"
        )
        if clustering not in available:
            raise KeyError(
                f"no clustering {clustering} in the cluster combo table; "
                f"available: {sorted(available)}"
            )
        resolved["leiden_resolution"] = _resolve_resolution(
            resolved["leiden_resolution"], resolved, optimal_resolutions
        )
        expanded.append(resolved)

    return (
        pd.DataFrame(expanded, columns=columns).drop_duplicates().reset_index(drop=True)
    )


def panel_tier_clusterings(screen, config, cluster_combos, split_by_compartment=False):
    """Return the clustering rows the lab's panel policy annotates for a screen.

    Every full panel, every grouped panel, every single marker and the DNA-only
    baseline are annotated; a DNA-plus-one-marker combo is dropped because the
    grouped panel containing those channels already covers its clusters. The
    rows carry no resolution, so each inherits the notebook's selection.

    Args:
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        cluster_combos (pd.DataFrame): The cluster phase's combo table.
        split_by_compartment (bool, optional): Whether compartment-specific
            paths are enabled. Defaults to False.

    Returns:
        list: Row dicts ready to hand to ``expand_mozzarellm_selection`` as
        ``extra_clusterings``.
    """
    columns = [
        key
        for key in _selection_columns(split_by_compartment)
        if key != "leiden_resolution"
    ]
    clusterings = cluster_combos[
        [column for column in columns if column in cluster_combos.columns]
    ].drop_duplicates()

    return [
        row
        for row in clusterings.to_dict("records")
        if panel_tier(screen, config, row["channel_combo"]) in PANEL_TIERS_ANNOTATED
    ]


def panel_tier(screen, config, channel_combo):
    """Return the panel tier a channel combo falls in.

    Args:
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        channel_combo (str): Underscore-joined channels the clustering used.

    Returns:
        str: One of ``full panel``, ``grouped panel``, ``single marker``,
        ``DNA plus one marker`` or ``DNA-only baseline``.
    """
    panel = _channel_names(screen, config)
    channels = _combo_channels(screen, config, channel_combo)
    markers = [channel for channel in channels if channel != _dna_channel(screen)]
    if not markers:
        return "DNA-only baseline"
    if len(markers) == 1:
        return (
            "single marker" if len(markers) == len(channels) else "DNA plus one marker"
        )

    return "full panel" if len(channels) == len(panel) else "grouped panel"


def write_screen_contexts(selection, screen, config, context_dir, rewrite=False):
    """Write one screen context JSON per selected clustering.

    A context is derived once as a starting point and then belongs to the
    operator, so an existing file is left alone unless ``rewrite`` is set. Edit
    the JSON for a one-off; edit ``screen.yaml`` for anything that should follow
    a channel into every combo that contains it.

    Args:
        selection (pd.DataFrame): Rows from ``expand_mozzarellm_selection``.
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        context_dir (str | Path): Directory the contexts are written into.
        rewrite (bool, optional): Overwrite contexts that already exist.
            Defaults to False.

    Returns:
        tuple: ``(context_paths, derived, kept)`` -- every row's context path,
        the paths written from ``screen.yaml``, and the paths left as the
        operator's own.
    """
    context_dir = Path(context_dir)
    context_dir.mkdir(parents=True, exist_ok=True)

    context_paths, derived, kept = [], [], []
    for row in selection.to_dict("records"):
        metadata = {"channel_combo": row["channel_combo"]}
        if "compartment_combo" in row:
            metadata["compartment_combo"] = row["compartment_combo"]
        metadata["cell_class"] = row["cell_class"]
        metadata["leiden_resolution"] = row["leiden_resolution"]
        context_fp = context_dir / get_filename(metadata, "screen_context", "json")
        payload = json.dumps(
            screen_context_for_combo(
                screen,
                config,
                row["channel_combo"],
                row["cell_class"],
                row["leiden_resolution"],
            ),
            indent=2,
        )
        if context_fp.exists() and not rewrite:
            kept.append(str(context_fp))
        elif (
            not context_fp.exists() or context_fp.read_text(encoding="utf-8") != payload
        ):
            context_fp.write_text(payload, encoding="utf-8")
            derived.append(str(context_fp))
        else:
            derived.append(str(context_fp))
        context_paths.append(str(context_fp))

    return context_paths, derived, kept


def write_mozzarellm_combo_table(selection, context_paths, combo_fp):
    """Write the combo table the mozzarellm stage builds its DAG from.

    Args:
        selection (pd.DataFrame): Rows from ``expand_mozzarellm_selection``.
        context_paths (list): Each row's screen context path, in row order.
        combo_fp (str | Path): Table path, ex ``config/mozzarellm_combo.tsv``.

    Returns:
        pd.DataFrame: The table as written, with ``screen_context_fp`` added.
    """
    combos = selection.copy()
    combos["screen_context_fp"] = context_paths
    combos.to_csv(combo_fp, sep="\t", index=False)

    return combos


def screen_context_for_combo(
    screen, config, channel_combo, cell_class, leiden_resolution
):
    """Build the screen context for one clustering's channel combo and cell class.

    A clustering is built from one channel combo and one cell class, so the
    context it gets describes only those channels rather than the screen's full
    panel. Callers write one of these per row of the mozzarellm combo table.

    Args:
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        channel_combo (str): Underscore-joined channels the clustering used.
        cell_class (str): Cell class the clustering covers, or "all".
        leiden_resolution (int | float | str): Resolution being annotated.

    Returns:
        dict: Screen context ready to be written as JSON.
    """
    return screen_context_from_screen(
        screen,
        config,
        leiden_resolution,
        channel_combo=channel_combo,
        cell_class=cell_class,
    )


def screen_context_from_screen(
    screen, config, leiden_resolution, channel_combo=None, cell_class=None
):
    """Build the mozzarellm screen-context dict from screen.yaml and config.yml.

    Every key the mozzarellm template marks "required" is filled, falling back
    to a brieflow default whenever the screen description leaves a value null.

    Args:
        screen (dict): Parsed ``screen.yaml``.
        config (dict): Parsed ``config/config.yml``.
        leiden_resolution (int | float | str): Resolution being annotated.
        channel_combo (str, optional): Channels the clustering was built from.
            Defaults to None (the screen's full channel panel).
        cell_class (str, optional): Cell class the clustering covers. Defaults
            to None (unnamed, as for an "all" clustering).

    Returns:
        dict: Screen context ready to be written as JSON.
    """
    experiment = _section(screen, "experiment")
    library = _section(screen, "library")
    collection = _section(screen, "collection")
    cluster_config = _section(config, "cluster")

    channels = _combo_channels(screen, config, channel_combo)
    gene_selection = library.get("gene_selection") or "pooled sgRNA library"
    number_of_genes = library.get("number_of_genes")
    if number_of_genes:
        gene_selection = f"{gene_selection} ({number_of_genes} genes)"
    metric = cluster_config.get("phate_distance_metric") or "euclidean"
    resolutions = cluster_config.get("leiden_resolutions") or [leiden_resolution]

    return {
        "assay_type": experiment.get("assay") or "optical pooled screening",
        "target_phenotype": _target_phenotype(screen, channels, cell_class),
        "organism": experiment.get("organism") or "Homo sapiens",
        "organism_ontology_term_id": (
            str(experiment.get("organism_ontology_term_id"))
            if experiment.get("organism_ontology_term_id")
            else f"NCBITaxon:{DEFAULT_ORGANISM_ID}"
        ),
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
    return organism_id_from_context(_section(screen, "experiment"))


def _selection_columns(split_by_compartment):
    """Return the columns one selection row carries for the configured path mode."""
    columns = [
        column
        for column in SELECTION_COLUMNS
        if column != "compartment_combo" or split_by_compartment
    ]

    return columns + ["leiden_resolution"]


def _expand_entry(entry, selection, columns):
    """Expand one extra-clustering entry into rows, inheriting the keys it omits."""
    unknown = set(entry) - set(columns)
    if unknown:
        raise KeyError(f"unknown mozzarellm selection keys {sorted(unknown)}")

    values = [
        value if isinstance(value, (list, tuple)) else [value]
        for value in entry.values()
    ]

    return [
        dict(selection, **dict(zip(entry, combination)))
        for combination in product(*values)
    ]


def _available_clusterings(cluster_combos, split_by_compartment):
    """Return the clusterings the cluster phase produced, as identity tuples."""
    columns = [
        column
        for column in _selection_columns(split_by_compartment)
        if column != "leiden_resolution"
    ]

    return {
        tuple(str(row[column]) for column in columns)
        for row in cluster_combos.to_dict("records")
    }


def _resolve_resolution(leiden_resolution, row, optimal_resolutions):
    """Resolve a row's resolution, looking up the benchmark pick only when asked."""
    if str(leiden_resolution) != OPTIMAL_RESOLUTION:
        return leiden_resolution

    key = f"{row['cell_class']}_{row['channel_combo']}"
    optimal = (optimal_resolutions or {}).get(key, {}).get("optimal_resolution")
    if optimal is None:
        raise KeyError(f"no benchmark-derived optimal resolution for {key}")

    return optimal


def _section(mapping, key):
    """Return a nested mapping section, treating a missing or null section as empty."""
    return (mapping or {}).get(key) or {}


def _dna_channel(screen):
    """Return the channel screen.yaml names as the nuclear marker."""
    phenotype = _section(screen, "phenotype")

    return phenotype.get("background_channel_nucleus") or DEFAULT_DNA_CHANNEL


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


def _combo_channels(screen, config, channel_combo):
    """Return the channels a clustering was built from, else the screen's full panel.

    A channel name can itself hold an underscore, so the combo is matched
    against the screen's own channel names as whole names rather than split on
    the delimiter; the split is the fallback for a combo naming no known channel.
    """
    if not channel_combo:
        return _channel_names(screen, config)

    combo = str(channel_combo)
    parts = [part for part in combo.split(CHANNEL_COMBO_DELIMITER) if part]
    named = [
        channel
        for channel in _channel_names(screen, config)
        if _contains_channel(parts, channel)
    ]

    return named or parts


def _contains_channel(parts, channel):
    """Check whether a combo's underscore-separated parts spell out a channel name."""
    wanted = [part for part in str(channel).split(CHANNEL_COMBO_DELIMITER) if part]
    if not wanted:
        return False

    return any(
        parts[start : start + len(wanted)] == wanted
        for start in range(len(parts) - len(wanted) + 1)
    )


def _channel_descriptions(screen):
    """Map each phenotype channel's biological name to how it should be described.

    A channel's free-text ``description`` in screen.yaml is used verbatim, so a
    fact about what a channel actually measures follows it into every combo that
    contains it; ``marker_of`` is the fallback phrasing.
    """
    described = {}
    for channel in _section(screen, "phenotype").get("channels") or []:
        if not isinstance(channel, dict) or not channel.get("biological_name"):
            continue
        name = str(channel["biological_name"])
        description = str(channel.get("description") or "").strip()
        marker_of = str(channel.get("marker_of") or "").strip()
        described[name] = description or (f"marker of {marker_of}" if marker_of else "")

    return described


def _target_phenotype(screen, channels, cell_class):
    """Render the target-phenotype sentence for one channel combo and cell class."""
    descriptions = _channel_descriptions(screen)
    described = [
        f"{channel} ({descriptions[channel]})" if descriptions.get(channel) else channel
        for channel in channels
    ]
    channel_text = ", ".join(described) if described else "the phenotype"
    plural = "s" if len(described) > 1 else ""
    scope = (
        f" of {cell_class} cells"
        if cell_class and str(cell_class).lower() != "all"
        else ""
    )

    return (
        "Morphological cell phenotype — shape, texture and intensity features"
        f"{scope} measured in the {channel_text} imaging channel{plural}"
    )


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
