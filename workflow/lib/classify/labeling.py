"""This module provides functions for manual cell classification UI.

Includes utilities for loading and thresholding mask data, selecting batches
for labeling, rendering images with overlays, and consolidating manual classifications
into training datasets.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import clear_output, display

from lib.classify.shared import (
    compose_rgb_crops,
    compute_crop_bounds,
    load_aligned_stack,
    load_mask_labels,
    overlay_mask_boundary_inplace,
    overlay_scale_bar,
    to_png_bytes,
    well_for_filename,
)
from lib.shared.file_utils import get_filename


def select_next_batch_from_pools(
    in_pool_df: pd.DataFrame,
    out_pool_df: pd.DataFrame,
    selection_mode: str,
    batch_size: int,
    *,
    keys: List[str],
    summary_df: pd.DataFrame | None = None,
    out_randomizer: int = 0,
    prioritized_in_keys: Set[Tuple[int, str, int, int]] | None = None,
    prioritized_out_keys: Set[Tuple[int, str, int, int]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Select the next batch of masks from in-/out-of-threshold pools with optional prioritization.

    Args:
        in_pool_df: In-gate masks.
        out_pool_df: Out-of-gate masks.
        selection_mode: 'random' or 'top_n'.
        batch_size: Total number of rows to select.
        keys: Key columns (e.g., ['plate','well','tile','mask_label']).
        summary_df: Tile counts (required for 'top_n').
        out_randomizer: Maximum number of out-of-threshold rows to include in a batch.
        prioritized_in_keys: Set of key tuples to prioritize from the in-pool.
        prioritized_out_keys: Set of key tuples to prioritize from the out-pool.

    Returns:
        (batch_df, debug_info) where batch_df contains only the key columns.
    """
    if out_randomizer > batch_size:
        raise ValueError(
            f"OUT_OF_GATE_COUNT ({out_randomizer}) cannot exceed BATCH_SIZE ({batch_size})."
        )

    prioritized_in_keys = prioritized_in_keys or set()
    prioritized_out_keys = prioritized_out_keys or set()

    total = len(in_pool_df) + len(out_pool_df)
    if total == 0:
        raise ValueError("No masks remaining to display.")

    show_n = min(int(batch_size), total)
    out_randomizer = max(0, min(int(out_randomizer), show_n))

    n_out = min(out_randomizer, len(out_pool_df))
    n_in = show_n - n_out

    def _split(df, pkeys):
        if df.empty or not pkeys:
            return df.iloc[0:0], df
        keydf = pd.DataFrame(list(pkeys), columns=keys)
        tagged = df.merge(keydf.assign(_p=1), on=keys, how="left")
        pr = tagged[tagged["_p"].notna()].drop(columns="_p")
        rg = tagged[tagged["_p"].isna()].drop(columns="_p")
        return pr.reset_index(drop=True), rg.reset_index(drop=True)

    in_pr, in_rg = _split(in_pool_df, prioritized_in_keys)
    out_pr, out_rg = _split(out_pool_df, prioritized_out_keys)

    def _pick(df, n):
        if n <= 0 or df.empty:
            return df.iloc[0:0]
        if selection_mode == "random":
            return df.sample(n=min(n, len(df)), replace=False, random_state=None)
        return _rank_by_tile(df, summary_df).head(n)

    in_sel_pr = _pick(in_pr, min(n_in, len(in_pr)))
    in_rem = n_in - len(in_sel_pr)
    in_sel_rg = _pick(in_rg, in_rem) if in_rem > 0 else in_rg.iloc[0:0]
    in_sel = pd.concat([in_sel_pr, in_sel_rg], ignore_index=True)

    out_sel_pr = _pick(out_pr, min(n_out, len(out_pr)))
    out_rem = n_out - len(out_sel_pr)
    out_sel_rg = _pick(out_rg, out_rem) if out_rem > 0 else out_rg.iloc[0:0]
    out_sel = pd.concat([out_sel_pr, out_sel_rg], ignore_index=True)

    got = len(in_sel) + len(out_sel)
    if got < show_n:
        short = show_n - got
        extra_in = _pick(in_rg.drop(in_sel_rg.index, errors="ignore"), short)
        need = short - len(extra_in)
        extra_out = (
            _pick(out_rg.drop(out_sel_rg.index, errors="ignore"), need)
            if need > 0
            else out_rg.iloc[0:0]
        )
        in_sel = pd.concat([in_sel, extra_in], ignore_index=True)
        out_sel = pd.concat([out_sel, extra_out], ignore_index=True)

    batch = pd.concat([in_sel[keys], out_sel[keys]], ignore_index=True)
    dbg = {
        "displaying": len(batch),
        "selected_in": len(in_sel),
        "selected_out": len(out_sel),
        "selected_in_prioritized": len(in_sel_pr),
        "selected_out_prioritized": len(out_sel_pr),
        "in_pool_remaining_before": len(in_pool_df),
        "out_pool_remaining_before": len(out_pool_df),
        "out_randomizer_requested": out_randomizer,
        "mode": selection_mode,
    }
    return batch, dbg


def remove_seen_from_pools(
    in_pool_df: pd.DataFrame,
    out_pool_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    *,
    keys: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove already-seen rows from both in-/out-of-threshold pools.

    Args:
        in_pool_df: In-gate masks pool.
        out_pool_df: Out-of-gate masks pool.
        seen_df: DataFrame containing rows that were displayed (with key columns).
        keys: Key columns (e.g., ['plate','well','tile','mask_label']).

    Returns:
        (in_left, out_left) with previously seen rows removed.
    """
    if seen_df.empty:
        return in_pool_df, out_pool_df
    seen = seen_df[keys].assign(__seen__=True)
    in_left = in_pool_df.merge(seen, on=keys, how="left")
    out_left = out_pool_df.merge(seen, on=keys, how="left")
    in_left = (
        in_left[in_left["__seen__"].isna()]
        .drop(columns="__seen__")
        .reset_index(drop=True)
    )
    out_left = (
        out_left[out_left["__seen__"].isna()]
        .drop(columns="__seen__")
        .reset_index(drop=True)
    )
    return in_left, out_left


# ----------------------------------------------------------------------------- #
# Consolidation (build training parquet)                                         #
# ----------------------------------------------------------------------------- #


def consolidate_manual_classifications(
    manual_classified_df: pd.DataFrame,
    class_title: str,
    mode: str,
    phenotype_output_fp: Path,
    classifier_output_dir: Path,
    timestamp: str | None = None,
    write: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a training parquet from a manual classification table.

    Args:
        manual_classified_df: Table with at least ['plate','well','tile','mask_label', class_title].
        class_title: Name of the classification column to write.
        mode: 'cell' or 'vacuole'.
        phenotype_output_fp: Root path containing 'parquets'.
        classifier_output_dir: Root path where 'training_dataset' is created.
        timestamp: Optional timestamp string used in output filename.
        write: If True, write the parquet to disk.
        verbose: If True, print progress messages.

    Returns:
        The consolidated training DataFrame (also written to disk if write=True).
    """
    if manual_classified_df is None or len(manual_classified_df) == 0:
        raise ValueError(
            "No manual classifications provided (manual_classified_df is empty)."
        )

    req_cols = ["plate", "well", "tile", "mask_label", class_title]
    missing_cols = [c for c in req_cols if c not in manual_classified_df.columns]
    if missing_cols:
        raise ValueError(
            f"manual_classified_df is missing required columns: {missing_cols}"
        )

    man_df = manual_classified_df.dropna(subset=req_cols).copy()
    if man_df.empty:
        raise ValueError(
            "All manual classification rows have missing values in required columns."
        )

    man_df["tile"] = pd.to_numeric(man_df["tile"], errors="coerce")
    man_df["mask_label"] = pd.to_numeric(man_df["mask_label"], errors="coerce")
    man_df[class_title] = pd.to_numeric(man_df[class_title], errors="coerce").astype(
        "Int64"
    )
    man_df["well"] = man_df["well"].astype(str)
    man_df = man_df.dropna(subset=["tile", "mask_label", class_title])
    man_df = man_df.drop_duplicates(
        subset=["plate", "well", "tile", "mask_label"], keep="last"
    )

    if man_df.empty:
        raise ValueError("No valid manual classifications remain after cleaning/dedup.")

    parquet_dir = _get_parquet_dir(phenotype_output_fp)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    selected_rows: list[pd.DataFrame] = []

    for (plate_val, well_val), grouped in man_df.groupby(["plate", "well"], sort=True):
        plate_int = int(plate_val)
        well_str = str(well_val)

        if mode == "vacuole":
            pq_path = parquet_dir / get_filename(
                {"plate": plate_int, "well": well_for_filename(well_str)},
                "phenotype_vacuoles",
                "parquet",
            )
            id_col = "vacuole_id"
        else:
            # Prefer 'phenotype_cp', fall back to 'phenotype_cp_min' if needed
            main = parquet_dir / get_filename(
                {"plate": plate_int, "well": well_for_filename(well_str)},
                "phenotype_cp",
                "parquet",
            )
            alt = parquet_dir / get_filename(
                {"plate": plate_int, "well": well_for_filename(well_str)},
                "phenotype_cp_min",
                "parquet",
            )
            pq_path = main if main.exists() else alt
            if not pq_path.exists():
                raise FileNotFoundError(f"Missing parquet: {main} (also tried {alt})")
            try:
                cols_schema = pq.ParquetFile(pq_path).schema.names
            except Exception:
                cols_schema = list(pd.read_parquet(pq_path).head(0).columns)
            id_col = (
                "label"
                if "label" in cols_schema
                else ("labels" if "labels" in cols_schema else None)
            )
            if id_col is None:
                df_tmp = pd.read_parquet(pq_path)
                id_col = (
                    "label"
                    if "label" in df_tmp.columns
                    else ("labels" if "labels" in df_tmp.columns else None)
                )
                if id_col is None:
                    raise KeyError(f"Neither 'label' nor 'labels' found in {pq_path}")

        if not pq_path.exists():
            raise FileNotFoundError(f"Missing parquet: {pq_path}")

        df_pq = pd.read_parquet(pq_path)
        if "tile" not in df_pq.columns or id_col not in df_pq.columns:
            raise KeyError(
                f"Required columns not found in {pq_path.name}: 'tile' or '{id_col}'"
            )

        tile_num = pd.to_numeric(df_pq["tile"], errors="coerce").to_numpy()
        id_num = pd.to_numeric(df_pq[id_col], errors="coerce").to_numpy()

        for _, r in grouped.iterrows():
            t = float(r["tile"])
            m = float(r["mask_label"])
            pos = np.where((tile_num == t) & (id_num == m))[0]
            if len(pos) == 0:
                raise ValueError(
                    f"No match in {pq_path.name} for tile={r['tile']} {id_col}={r['mask_label']}"
                )
            if len(pos) > 1:
                raise ValueError(
                    f"Multiple matches in {pq_path.name} for tile={r['tile']} {id_col}={r['mask_label']}"
                )
            row_df = df_pq.iloc[[pos[0]]].copy()
            row_df[class_title] = int(r[class_title])
            selected_rows.append(row_df)

    if not selected_rows:
        raise ValueError(
            "No rows were selected. Check manual classifications and source parquets."
        )

    consolidated_df = pd.concat(selected_rows, ignore_index=True)
    if class_title in consolidated_df.columns:
        cols = [c for c in consolidated_df.columns if c != class_title] + [class_title]
        consolidated_df = consolidated_df[cols]

    if write:
        train_dir = Path(classifier_output_dir) / "training_dataset"
        train_dir.mkdir(parents=True, exist_ok=True)
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = (
            f"{mode}_classifier_training_dataset_for_{class_title}_{timestamp}.parquet"
        )
        out_path = train_dir / out_name
        consolidated_df.to_parquet(out_path, index=False)
        if verbose:
            print(f"Saved {len(consolidated_df)} labeled rows to: {out_path}")
    else:
        if verbose:
            print(
                f"Built consolidated dataset with {len(consolidated_df)} rows (write=False)."
            )

    return consolidated_df, out_path if write else None


# ----------------------------------------------------------------------------- #
# Mask dataframe preparation (standalone callable)                               #
# ----------------------------------------------------------------------------- #


def prepare_mask_dataframes(
    *,
    mode: str,
    phenotype_fp: Union[str, Path],
    plates: Sequence[Union[str, int]],
    wells: Sequence[Union[str, int, str]],
    keys: Sequence[str] = ("plate", "well", "tile", "mask_label"),
    gate_feature: Optional[Union[str, Sequence[Optional[str]]]] = None,
    gate_min: Optional[Union[float, Sequence[Optional[float]]]] = None,
    gate_max: Optional[Union[float, Sequence[Optional[float]]]] = None,
    gate_min_percentile: Optional[Union[float, Sequence[Optional[float]]]] = None,
    gate_max_percentile: Optional[Union[float, Sequence[Optional[float]]]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Build mask summary and key tables from per-well parquets and (optionally) apply thresholds.

    Args:
        mode: 'vacuole' or 'cell'.
        phenotype_fp: Root phenotype output path (contains 'parquets' subdirectory).
        plates: Iterable of plate identifiers to include.
        wells: Iterable of well identifiers to include.
        keys: Key column names (default ['plate','well','tile','mask_label']).
        gate_feature: Feature name or list of names to filter by (None for no filtering).
        gate_min: Numeric lower bound(s) (exclusive) or None.
        gate_max: Numeric upper bound(s) (exclusive) or None.
        gate_min_percentile: Lower quantile(s) in [0,1] (exclusive) or None.
        gate_max_percentile: Upper quantile(s) in [0,1] (exclusive) or None.
        verbose: If True, print progress messages.

    Returns:
        (summary_df, in_gate_df_in, in_gate_df_out, gate_debug)
    """
    pq_dir = _get_parquet_dir(phenotype_fp)
    if not pq_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {pq_dir}")

    plate_set = {str(p) for p in plates}
    well_set = {str(w) for w in wells}

    tile_rows: List[pd.DataFrame] = []
    instance_rows: List[pd.DataFrame] = []

    for plate in sorted(plate_set):
        for well in sorted(well_set):
            pq_path = _pq_path_for(plate, well, phenotype_fp, mode)
            if not pq_path.exists():
                if verbose:
                    print(f"[warn] Skipping missing parquet: {pq_path}")
                continue

            try:
                cols_schema = pq.ParquetFile(pq_path).schema.names
            except Exception:
                cols_schema = list(pd.read_parquet(pq_path).head(0).columns)
            id_col = _id_col_for_mode(cols_schema, mode)

            df = pd.read_parquet(pq_path, columns=["tile", id_col]).dropna()
            df["tile"] = pd.to_numeric(df["tile"], errors="coerce")
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
            df = df.dropna()
            df["tile"] = df["tile"].astype(int)
            df[id_col] = df[id_col].astype(int)
            df = df.loc[df[id_col] > 0, ["tile", id_col]].drop_duplicates()

            if not df.empty:
                tile_counts = (
                    df.groupby("tile", as_index=False)
                    .size()
                    .rename(columns={"size": "number_of_masks"})
                )
                tile_counts.insert(0, "plate", int(plate))
                tile_counts.insert(1, "well", str(well))
                tile_rows.append(tile_counts)

                inst = df.rename(columns={id_col: "mask_label"})[
                    ["tile", "mask_label"]
                ].copy()
                inst.insert(0, "plate", int(plate))
                inst.insert(1, "well", str(well))
                instance_rows.append(inst)

    if tile_rows:
        summary_df = (
            pd.concat(tile_rows, ignore_index=True)
            .sort_values("number_of_masks", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        summary_df = pd.DataFrame(columns=["plate", "well", "tile", "number_of_masks"])

    if instance_rows:
        in_gate_df_all = (
            pd.concat(instance_rows, ignore_index=True)
            .sort_values(list(keys))
            .reset_index(drop=True)
        )
    else:
        in_gate_df_all = pd.DataFrame(columns=list(keys))

    if verbose:
        print(
            f"[parquet] Tiles: {len(summary_df)} | Masks: {len(in_gate_df_all)} (mode={mode})"
        )

    # Early guard: if no masks were loaded at all, provide a clearer error message
    if in_gate_df_all.empty:
        raise ValueError(
            "No masks were found for the selected plates/wells and mode.\n"
            f"- mode: {mode}\n"
            f"- plates: {sorted(plate_set)}\n"
            f"- wells: {sorted(well_set)}\n"
        )

    # normalize threshold specs
    def _normalize_threshold_specs() -> List[Dict]:
        feats = gate_feature
        if feats is None:
            return []
        feats = feats if isinstance(feats, (list, tuple, np.ndarray)) else [feats]
        L = len(feats)
        mins = _as_list(gate_min, L, "gate_min")
        maxs = _as_list(gate_max, L, "gate_max")
        minpct = _as_list(gate_min_percentile, L, "gate_min_percentile")
        maxpct = _as_list(gate_max_percentile, L, "gate_max_percentile")

        specs: List[Dict] = []
        for i in range(L):
            feat = feats[i]
            if feat is None:
                continue
            sp = {
                "feature": feat,
                "min_num": mins[i],
                "max_num": maxs[i],
                "min_pct": minpct[i],
                "max_pct": maxpct[i],
            }
            # skip pure no-op specs
            if (
                sp["min_num"] is None
                and sp["max_num"] is None
                and sp["min_pct"] is None
                and sp["max_pct"] is None
            ):
                continue
            specs.append(sp)
        return specs

    specs = _normalize_threshold_specs()
    if not specs:
        in_gate_df = in_gate_df_all.copy()
        out_of_gate_df = in_gate_df_all.iloc[0:0].copy()
        gate_debug: List[Dict] = []
        if verbose:
            print("No thresholding (no usable bounds provided).")
    else:
        in_gate_df, out_of_gate_df, gate_debug = _apply_multi_thresholds(
            in_gate_df_all=in_gate_df_all,
            mode=mode,
            pq_dir=pq_dir,
            specs=specs,
            _KEYS=list(keys),
        )
        if verbose:
            print("Thresholding applied (multi-filter intersection):")
            for d in gate_debug:
                print(
                    f"  [#{d['filter_index']}] {d['feature']}: "
                    f"min(>): {d['min_open']}, max(<): {d['max_open']} | "
                    f"global[{d['global_min']}, {d['global_max']}] | "
                    f"kept: {d['kept_after_this_filter']}"
                )
            print(f"\nin_gate_df (IN): {len(in_gate_df)} rows")
            print(f"out_of_gate_df (OUT): {len(out_of_gate_df)} rows")

    summary_df = _update_mask_summary(summary_df, in_gate_df, out_of_gate_df)

    if verbose:
        print("\nFinal tables ready for the UI:")
        print(f"  summary_df: {len(summary_df)} tiles")
        print(f"  in_gate_df (IN): {len(in_gate_df)} rows")
        print(f"  out_of_gate_df (OUT): {len(out_of_gate_df)} rows")

    return (
        summary_df,
        in_gate_df,
        out_of_gate_df,
        gate_debug,
    )


# ----------------------------------------------------------------------------- #
# Core helpers                                                                  #
# ----------------------------------------------------------------------------- #


def build_class_mapping(classification: List[str]) -> Dict:
    """Build a class mapping dictionary from a list of class names.

    Args:
        classification: List of class names in order (e.g., ["1 parasite", "2-3 parasite"]).

    Returns:
        Dict with 'label_to_class' mapping 1-based indices to class names.
    """
    return {"label_to_class": {i + 1: label for i, label in enumerate(classification)}}


def to_list_of_str(value, fallback=None) -> List[str]:
    """Normalize a value to a list of strings.

    Args:
        value: Value to normalize (Path, str, list, tuple, set, or None).
        fallback: Fallback value if value is None.

    Returns:
        List of strings.
    """
    base = fallback if value is None else value
    if base is None:
        return []
    if isinstance(base, (list, tuple, set)):
        return [str(x) for x in base]
    return [str(base)]


def resolve_channel_colors(
    display_channels: List[str],
    channel_colors: List[str],
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Resolve channel color names to RGB tuples.

    Args:
        display_channels: List of channel names to display.
        channel_colors: List of matplotlib color names or hex codes.

    Returns:
        List of (type, rgb) tuples where type is 'gray' or 'rgb'.
    """
    import matplotlib.colors as mcolors

    resolved = []
    for idx, ch in enumerate(display_channels):
        color_name = channel_colors[idx] if idx < len(channel_colors) else None
        if color_name is None:
            resolved.append(("gray", (1.0, 1.0, 1.0)))
        else:
            try:
                rgb = mcolors.to_rgb(color_name)
                resolved.append(("rgb", rgb))
            except ValueError:
                raise ValueError(
                    f"Invalid color '{color_name}' for channel '{ch}'. "
                    "Use a valid matplotlib color name or hex."
                )
    return resolved


def load_existing_training_data(
    existing_training_path: Union[str, Path],
    mode: str,
    class_title: str,
) -> Tuple[pd.DataFrame, set]:
    """Load existing training data and extract keys.

    Args:
        existing_training_path: Path to existing training parquet.
        mode: Classification mode ('cell' or 'vacuole').
        class_title: Name of the classification column.

    Returns:
        (normalized_df, existing_keys_set)
    """
    df_existing = pd.read_parquet(existing_training_path)
    seeded = _normalize_keys(df_existing, mode, class_title)
    seeded["_existing"] = True

    existing_keys = set(
        (int(r.plate), str(r.well), int(r.tile), int(r.mask_label))
        for r in seeded.itertuples(index=False)
    )
    return seeded, existing_keys


def filter_existing_from_pools(
    in_gate_df: pd.DataFrame,
    out_of_gate_df: pd.DataFrame,
    existing_keys: set,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove existing training keys from gate pools.

    Args:
        in_gate_df: In-gate pool DataFrame.
        out_of_gate_df: Out-of-gate pool DataFrame.
        existing_keys: Set of (plate, well, tile, mask_label) tuples.

    Returns:
        (filtered_in_gate_df, filtered_out_of_gate_df)
    """
    if not existing_keys:
        return in_gate_df, out_of_gate_df

    keys_df = pd.DataFrame(
        list(existing_keys), columns=["plate", "well", "tile", "mask_label"]
    ).assign(_ex=1)

    if not in_gate_df.empty:
        in_gate_df = in_gate_df.merge(
            keys_df, on=["plate", "well", "tile", "mask_label"], how="left"
        )
        in_gate_df = (
            in_gate_df[in_gate_df["_ex"].isna()]
            .drop(columns="_ex")
            .reset_index(drop=True)
        )

    if not out_of_gate_df.empty:
        out_of_gate_df = out_of_gate_df.merge(
            keys_df, on=["plate", "well", "tile", "mask_label"], how="left"
        )
        out_of_gate_df = (
            out_of_gate_df[out_of_gate_df["_ex"].isna()]
            .drop(columns="_ex")
            .reset_index(drop=True)
        )

    return in_gate_df, out_of_gate_df


def initialize_labeling_state(
    random_seed: int,
    mode: str,
    class_title: str,
    keys: List[str],
    existing_classified_df: pd.DataFrame = None,
    existing_unclassified_df: pd.DataFrame = None,
) -> dict:
    """Initialize the state dictionary for the labeling UI.

    Args:
        random_seed: Random seed for reproducibility.
        mode: Classification mode ('cell' or 'vacuole').
        class_title: Name of the classification column.
        keys: Key column names (e.g., ['plate','well','tile','mask_label']).
        existing_classified_df: Optional existing classified dataframe.
        existing_unclassified_df: Optional existing unclassified dataframe.

    Returns:
        Initialized state dictionary.
    """
    state = {
        "rng": np.random.default_rng(random_seed),
        "aligned_cache": {},
        "mask_cache": {},
        "parquet_cache": {},
        "mode": mode,
        "container": None,
        "rows_state": [],
        "button": None,
        "status": None,
        "channel_header": None,
        "tile_order_df": None,
        "tile_idx": 0,
    }

    if existing_classified_df is not None and not existing_classified_df.empty:
        state["manual_classified_df"] = existing_classified_df.copy()
    else:
        state["manual_classified_df"] = None

    if existing_unclassified_df is not None and not existing_unclassified_df.empty:
        state["manual_unclassified_df"] = existing_unclassified_df.copy()
    else:
        state["manual_unclassified_df"] = pd.DataFrame(columns=keys)

    state["manual_classified_df"] = _ensure_mc_schema(
        state["manual_classified_df"], class_title, keys
    )

    return state


def _get_parquet_dir(base_path: Union[str, Path]) -> Path:
    """Get the parquet directory, avoiding double 'parquets' in path.

    Args:
        base_path: Base phenotype output path.

    Returns:
        Path to the parquets directory.
    """
    p = Path(base_path)
    if p.name == "parquets":
        return p
    return p / "parquets"


def _id_col_for_mode(columns, mode: str) -> str:
    """Determine the ID column name used inside phenotype parquet files for a mode.

    Args:
        columns: Iterable of column names present in the parquet.
        mode: Normalized mode ('vacuole' or 'cell').

    Returns:
        The ID column name to use ('vacuole_id' | 'label' | 'labels').
    """
    cols = set(columns)
    if mode == "vacuole":
        if "vacuole_id" in cols:
            return "vacuole_id"
        raise ValueError("Parquet is missing 'vacuole_id' for vacuole mode.")
    for c in ("label", "labels"):
        if c in cols:
            return c
    raise ValueError("Parquet is missing 'label'/'labels' for cell mode.")


def _pq_path_for(plate, well, phenotype_fp: Path, mode: str) -> Path:
    """Construct the phenotype parquet path for a given plate, well, and mode.

    Args:
        plate: Plate identifier.
        well: Well identifier.
        phenotype_fp: Phenotype output directory (parquets subdirectory is appended).
        mode: 'cell' or 'vacuole'.

    Returns:
        The resolved parquet file path. For cell mode, if 'phenotype_cp.parquet' is
        missing, falls back to 'phenotype_cp_min.parquet' when present.
    """
    pq_dir = _get_parquet_dir(phenotype_fp)
    wnorm = well_for_filename(well)
    if mode == "vacuole":
        return pq_dir / get_filename(
            {"plate": plate, "well": wnorm}, "phenotype_vacuoles", "parquet"
        )
    # cell mode: try cp then cp_min
    main = pq_dir / get_filename(
        {"plate": plate, "well": wnorm}, "phenotype_cp", "parquet"
    )
    if main.exists():
        return main
    alt = pq_dir / get_filename(
        {"plate": plate, "well": wnorm}, "phenotype_cp_min", "parquet"
    )
    return alt


def _quantile(series: pd.Series, q: float) -> float:
    """Compute the q-th quantile of a numeric Series.

    Args:
        series: Numeric pandas Series.
        q: Quantile in [0, 1].

    Returns:
        The computed quantile as a float.
    """
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError("Percentiles must be within [0, 1].")
    return float(
        np.quantile(pd.to_numeric(series, errors="coerce").dropna().to_numpy(), q)
    )


def _compute_bounds(
    series: pd.Series, min_num, max_num, min_pct, max_pct
) -> tuple[float, float, dict]:
    """Compute numeric lower/upper bounds for thresholding a feature series.

    Args:
        series: Numeric pandas Series of feature values.
        min_num: Optional numeric lower bound (inclusive if provided).
        max_num: Optional numeric upper bound (exclusive if provided).
        min_pct: Optional lower bound as quantile in [0, 1] (inclusive if provided).
        max_pct: Optional upper bound as quantile in [0, 1] (exclusive if provided).

    Returns:
        (min_val, max_val, stats) where stats contains 'global_min' and 'global_max'.
    """
    ser = pd.to_numeric(series, errors="coerce").dropna()
    if ser.empty:
        raise ValueError(
            "Feature series is empty or non-numeric; cannot compute thresholds."
        )

    gmin, gmax = float(ser.min()), float(ser.max())
    stats = {"global_min": gmin, "global_max": gmax}

    mins, maxs = [], []
    if min_num is not None:
        mins.append(float(min_num))
    if min_pct is not None:
        mins.append(_quantile(ser, float(min_pct)))
    if max_num is not None:
        maxs.append(float(max_num))
    if max_pct is not None:
        maxs.append(_quantile(ser, float(max_pct)))

    # If a side is unspecified, nudge global bound to make it inclusive with open-interval filter.
    min_val = min(mins) if mins else np.nextafter(gmin, -np.inf)
    max_val = max(maxs) if maxs else np.nextafter(gmax, np.inf)

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError("Non-finite thresholds computed; check inputs.")
    if min_val > max_val:
        raise ValueError(f"Invalid thresholds: min ({min_val}) > max ({max_val}).")

    return float(min_val), float(max_val), stats


def _as_list(x, L, name):
    """Broadcast a scalar (or None) to a list of length L, or validate a list-like length.

    Args:
        x: Scalar/None or list-like.
        L: Desired length.
        name: Parameter name (for diagnostics).

    Returns:
        A Python list of length L.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != L:
            raise ValueError(f"{name} must have length {L}, got {len(x)}.")
        return list(x)
    return [x] * L


def _rank_by_tile(df: pd.DataFrame, summary_df: pd.DataFrame | None) -> pd.DataFrame:
    """Rank masks by tile density (tiles with more masks first) and then by key order.

    Args:
        df: DataFrame with columns ['plate','well','tile','mask_label'].
        summary_df: DataFrame with columns ['plate','well','tile','number_of_masks'].

    Returns:
        A DataFrame sorted by tile rank and then ['plate','well','tile','mask_label'].
    """
    if summary_df is None or summary_df.empty or df.empty:
        return df
    ranked = summary_df.sort_values(
        "number_of_masks", ascending=False, kind="mergesort"
    )
    rank_map = {
        (int(r.plate), str(r.well), int(r.tile)): i
        for i, r in enumerate(ranked.itertuples(index=False))
    }
    out = df.copy()
    out["__rank__"] = out.apply(
        lambda r: rank_map.get(
            (int(r["plate"]), str(r["well"]), int(r["tile"])), 1_000_000
        ),
        axis=1,
    )
    out = out.sort_values(
        ["__rank__", "plate", "well", "tile", "mask_label"], kind="mergesort"
    )
    return out.drop(columns="__rank__")


# ----------------------------------------------------------------------------- #
# Thresholding helpers (feature index + multi-filter)                            #
# ----------------------------------------------------------------------------- #


def _load_feature_index(
    in_gate_df_all: pd.DataFrame, mode: str, pq_dir: Path, feature: str
) -> pd.DataFrame:
    """Load feature values from per-well parquets for rows present in a mask pool.

    Args:
        in_gate_df_all: Pool with key columns ['plate','well','tile','mask_label'].
        mode: Normalized mode ('vacuole' or 'cell').
        pq_dir: Directory containing phenotype parquets.
        feature: Column to load as the feature.

    Returns:
        A DataFrame with columns ['plate','well','tile','mask_label','_feature_value'].
    """
    missing_feature = []
    frames = []

    for plate, well in (
        in_gate_df_all[["plate", "well"]].drop_duplicates().itertuples(index=False)
    ):
        wnorm = well_for_filename(well)
        if mode == "vacuole":
            pq_path = pq_dir / get_filename(
                {"plate": plate, "well": wnorm}, "phenotype_vacuoles", "parquet"
            )
        else:
            main = pq_dir / get_filename(
                {"plate": plate, "well": wnorm}, "phenotype_cp", "parquet"
            )
            alt = pq_dir / get_filename(
                {"plate": plate, "well": wnorm}, "phenotype_cp_min", "parquet"
            )
            pq_path = main if main.exists() else alt
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing parquet: {pq_path}")

        try:
            cols_schema = pq.ParquetFile(pq_path).schema.names
        except Exception:
            cols_schema = list(pd.read_parquet(pq_path).head(0).columns)
        id_col = _id_col_for_mode(cols_schema, mode)

        pq_df = pd.read_parquet(pq_path, columns=["tile", id_col, feature])
        if feature not in pq_df.columns:
            missing_feature.append(str(pq_path))
            continue

        keep_tiles = in_gate_df_all.query("plate == @plate and well == @well")[
            "tile"
        ].unique()
        sub = pq_df.loc[
            pq_df["tile"].isin(keep_tiles), ["tile", id_col, feature]
        ].copy()
        sub.rename(
            columns={id_col: "mask_label", feature: "_feature_value"}, inplace=True
        )
        sub.insert(0, "plate", plate)
        sub.insert(1, "well", well)
        frames.append(sub)

    if missing_feature:
        msg = "\n".join(f"- {p}" for p in missing_feature)
        raise ValueError(
            f"THRESHOLD_FEATURE '{feature}' not found in these parquets:\n{msg}"
        )

    if not frames:
        raise ValueError(
            "No rows loaded to build feature index (check inputs and pool)."
        )

    feat = pd.concat(frames, ignore_index=True)
    feat["_feature_value"] = pd.to_numeric(feat["_feature_value"], errors="coerce")
    feat.dropna(subset=["_feature_value"], inplace=True)

    # Normalize dtypes for merge keys
    for c in ("plate", "well", "tile", "mask_label"):
        if c in feat.columns and c in in_gate_df_all.columns:
            try:
                feat[c] = feat[c].astype(in_gate_df_all[c].dtype)
            except Exception:
                pass
    return feat


def _apply_multi_thresholds(
    in_gate_df_all: pd.DataFrame,
    mode: str,
    pq_dir: Path,
    specs: list[dict],
    _KEYS: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Apply one or more threshold filters (intersection) to a mask pool.

    Args:
        in_gate_df_all: Full pool of mask keys.
        mode: Normalized mode ('vacuole' or 'cell').
        pq_dir: Directory containing phenotype parquets.
        specs: List of filter dicts with keys: feature, min_num, max_num, min_pct, max_pct.
        _KEYS: Key column names (e.g., ['plate','well','tile','mask_label']).

    Returns:
        (in_df, out_df, debug_list) where in_df is the filtered pool, out_df is the complement.
    """
    if not specs:
        return in_gate_df_all.copy(), in_gate_df_all.iloc[0:0].copy(), []

    pool = in_gate_df_all.copy()
    debug = []

    for idx, sp in enumerate(specs, start=1):
        feat_idx = _load_feature_index(pool, mode, pq_dir, sp["feature"])
        min_val, max_val, stats = _compute_bounds(
            series=feat_idx["_feature_value"],
            min_num=sp["min_num"],
            max_num=sp["max_num"],
            min_pct=sp["min_pct"],
            max_pct=sp["max_pct"],
        )

        keyed = pool.merge(feat_idx, on=_KEYS, how="left", validate="one_to_one")
        if keyed["_feature_value"].isna().any():
            miss = keyed[keyed["_feature_value"].isna()][_KEYS]
            raise ValueError(
                f"[filter {idx}] Missing feature values (feature='{sp['feature']}').\n"
                + miss.head(15).to_string(index=False)
            )

        keep = (keyed["_feature_value"] > min_val) & (keyed["_feature_value"] < max_val)
        kept = keyed.loc[keep, _KEYS].reset_index(drop=True)
        if kept.empty:
            raise ValueError(
                f"[filter {idx}] Feature '{sp['feature']}' thresholds yielded 0 in-range masks."
            )

        debug.append(
            {
                "filter_index": idx,
                "feature": sp["feature"],
                "min_open": float(min_val),
                "max_open": float(max_val),
                "global_min": float(stats["global_min"]),
                "global_max": float(stats["global_max"]),
                "kept_after_this_filter": int(len(kept)),
            }
        )
        pool = kept

    marker = pool.assign(__in__=1)
    out_df = (
        in_gate_df_all.merge(marker, on=_KEYS, how="left")
        .loc[lambda d: d["__in__"].isna(), _KEYS]
        .reset_index(drop=True)
    )
    return pool, out_df, debug


def _update_mask_summary(
    summary_df: pd.DataFrame, in_df: pd.DataFrame, out_df: pd.DataFrame
) -> pd.DataFrame:
    """Add per-tile in/out counts to a mask summary table.

    Args:
        summary_df: Base summary with ['plate','well','tile','number_of_masks'].
        in_df: In-gate mask keys.
        out_df: Out-of-gate mask keys.

    Returns:
        A summary DataFrame with added columns:
        ['in_range','out_of_range','number_of_masks_in_gate','number_of_masks_out_of_gate'].
    """
    _ms = summary_df.copy()
    drop_cols = [
        c
        for c in (
            "in_range",
            "out_of_range",
            "number_of_masks_in_gate",
            "number_of_masks_out_of_gate",
        )
        if c in _ms.columns
    ]
    if drop_cols:
        _ms = _ms.drop(columns=drop_cols)

    counts_in = (
        (
            in_df.groupby(["plate", "well", "tile"])
            .size()
            .rename("in_range")
            .reset_index()
        )
        if not in_df.empty
        else pd.DataFrame(columns=["plate", "well", "tile", "in_range"])
    )
    counts_out = (
        (
            out_df.groupby(["plate", "well", "tile"])
            .size()
            .rename("out_of_range")
            .reset_index()
        )
        if not out_df.empty
        else pd.DataFrame(columns=["plate", "well", "tile", "out_of_range"])
    )

    _ms = _ms.merge(counts_in, on=["plate", "well", "tile"], how="left")
    _ms = _ms.merge(counts_out, on=["plate", "well", "tile"], how="left")
    _ms["in_range"] = _ms.get("in_range", 0).fillna(0).astype(int)
    _ms["out_of_range"] = _ms.get("out_of_range", 0).fillna(0).astype(int)
    _ms["number_of_masks_in_gate"] = _ms["in_range"]
    _ms["number_of_masks_out_of_gate"] = _ms["out_of_range"]
    return _ms


# ----------------------------------------------------------------------------- #
# Training table utilities                                                       #
# ----------------------------------------------------------------------------- #


def _normalize_keys(df: pd.DataFrame, mode: str, class_col: str) -> pd.DataFrame:
    """Normalize key dtypes and rename the mode-specific ID column to 'mask_label'.

    Args:
        df: Input DataFrame containing plate/well/tile, the ID column, and class_col.
        mode: Normalized mode ('vacuole' or 'cell').
        class_col: Name of the classification column.

    Returns:
        DataFrame with columns ['plate','well','tile','mask_label', class_col].
    """
    id_col = _id_col_for_mode(df.columns, mode)
    req = ["plate", "well", "tile", id_col, class_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Existing training parquet is missing columns: {missing}")

    out = df.loc[:, req].copy()
    out["plate"] = pd.to_numeric(out["plate"], errors="coerce").astype("Int64")
    out["tile"] = pd.to_numeric(out["tile"], errors="coerce").astype("Int64")
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    out[class_col] = pd.to_numeric(out[class_col], errors="coerce").astype("Int64")
    out["well"] = out["well"].astype(str)
    out = out.dropna(subset=["plate", "tile", id_col, class_col]).copy()
    out.rename(columns={id_col: "mask_label"}, inplace=True)
    out["plate"] = out["plate"].astype(int)
    out["tile"] = out["tile"].astype(int)
    out["mask_label"] = out["mask_label"].astype(int)
    return out[["plate", "well", "tile", "mask_label", class_col]]


def _ensure_mc_schema(
    df: pd.DataFrame | None, CLASS_TITLE: str, _KEYS: list
) -> pd.DataFrame:
    """Ensure manual_classified_df has expected columns and helper flags.

    Args:
        df: Existing manual_classified_df or None.
        CLASS_TITLE: Name of the classification column.
        _KEYS: Key column names (e.g., ['plate','well','tile','mask_label']).

    Returns:
        A DataFrame with _KEYS + [CLASS_TITLE, '_existing', '_sprinkle'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_KEYS + [CLASS_TITLE, "_existing", "_sprinkle"])
    out = df.copy()
    if CLASS_TITLE not in out.columns:
        raise ValueError(
            f"manual_classified_df exists but is missing '{CLASS_TITLE}'. "
            "Ensure CLASS_TITLE matches the seeded training column name."
        )
    if "_existing" not in out.columns:
        out["_existing"] = False
    if "_sprinkle" not in out.columns:
        out["_sprinkle"] = False
    for c in _KEYS:
        if c not in out.columns:
            out[c] = pd.Series(dtype="Int64" if c != "well" else "object")
    return out


# ----------------------------------------------------------------------------- #
# Image / UI helpers                                                             #
# ----------------------------------------------------------------------------- #


# Removed thin wrappers around shared functions; use shared utilities directly.


def _render_row(
    meta: Dict,
    *,
    state: dict,
    MODE: str,
    PHENOTYPE_OUTPUT_FP: Path,
    CHANNEL_NAMES: List[str],
    CHANNEL_INDICES: List[int],
    RESOLVED_COLORS: List[Tuple[str, Tuple[float, float, float]]],
    SCALE_BAR: int,
    DISPLAY_CHANNEL: List[str],
    CLASSIFICATION: List[str],
) -> widgets.Widget:
    """Render a single UI row showing channel crops and a merged image with a boundary overlay.

    Args:
        meta: Row metadata containing keys and optional flags.
        state: Mutable cache/state dict.
        MODE: Normalized mode ('vacuole' or 'cell').
        PHENOTYPE_OUTPUT_FP: Root output path.
        CHANNEL_NAMES: Channel names (for shape validation).
        CHANNEL_INDICES: Indices of channels to display.
        RESOLVED_COLORS: List of ('gray'|tag, (r,g,b)) tuples per channel.
        SCALE_BAR: Scale bar length in pixels (0 to disable).
        DISPLAY_CHANNEL: Labels shown above channel images.
        CLASSIFICATION: List of class names for the dropdown.

    Returns:
        A widgets.Widget representing the row.
    """
    plate, well, tile, mask_label = (
        meta["plate"],
        meta["well"],
        meta["tile"],
        meta["mask_label"],
    )

    stack = load_aligned_stack(
        PHENOTYPE_OUTPUT_FP,
        CHANNEL_NAMES,
        int(plate),
        str(well),
        int(tile),
        cache=state.get("aligned_cache"),
    )

    H, W = stack.shape[1], stack.shape[2]
    y0, y1, x0, x1 = compute_crop_bounds(
        PHENOTYPE_OUTPUT_FP,
        MODE,
        int(plate),
        str(well),
        int(tile),
        int(mask_label),
        (H, W),
        mask_cache=state.get("mask_cache"),
        parquet_cache=state.get("parquet_cache"),
    )

    imgs, merged = compose_rgb_crops(
        stack, y0, y1, x0, x1, CHANNEL_INDICES, RESOLVED_COLORS
    )

    labels_full = load_mask_labels(
        PHENOTYPE_OUTPUT_FP,
        MODE,
        int(plate),
        str(well),
        int(tile),
        cache=state.get("mask_cache"),
    )
    labels_crop = labels_full[y0:y1, x0:x1]
    mask_crop = labels_crop == mask_label
    if np.any(mask_crop):
        overlay_mask_boundary_inplace(merged, mask_crop, step=2, value=1.0)

    # scale bar (use shared overlay)
    if SCALE_BAR and SCALE_BAR > 0:
        overlay_scale_bar(merged, int(SCALE_BAR))

    # images -> widgets
    img_widgets = []
    for arr in imgs + [merged]:
        png = to_png_bytes(arr)
        iw = widgets.Image(value=png, format="png")
        iw.layout = widgets.Layout(width="200px", height="200px")
        img_widgets.append(iw)

    header = widgets.HTML(f"<b>P-{plate} W-{well} T-{tile} | mask {mask_label}</b>")
    pre_idx = meta.get("_prefill_class_idx", None)
    pre_value = (
        CLASSIFICATION[int(pre_idx) - 1]
        if (pre_idx is not None and 1 <= int(pre_idx) <= len(CLASSIFICATION))
        else "--select class--"
    )
    dd = widgets.Dropdown(
        options=["--select class--"] + CLASSIFICATION,
        value=pre_value,
        layout=widgets.Layout(width="220px"),
    )

    left_children = [header, dd]
    if meta.get("_existing", False):
        left_children.append(
            widgets.HTML(
                "<div style='color:#1565c0; font-weight:700; margin-top:6px;'>Existing training dataset</div>"
            )
        )
        if pre_idx is None or pre_value == "--select class--":
            left_children.append(
                widgets.HTML(
                    "<div style='color:#e65100; font-weight:600; margin-top:4px;'>Class not present in current CLASSIFICATION — please reselect</div>"
                )
            )
    if meta.get("_sprinkle", False):
        left_children.append(
            widgets.HTML(
                "<div style='color:#b00020; font-weight:700; margin-top:4px;'>Out of threshold image</div>"
            )
        )

    left = widgets.VBox(left_children, layout=widgets.Layout(width="260px"))
    right = widgets.HBox(img_widgets, layout=widgets.Layout(align_items="center"))
    row = widgets.HBox([left, right])

    state.setdefault("rows_state", [])
    state["rows_state"].append({"meta": meta, "dropdown": dd})
    return row


def _collect_and_advance_random(
    state: dict, CLASSIFICATION: List[str], CLASS_TITLE: str, keys: List[str]
) -> None:
    """Collect selected labels from the current UI page and update state tables.

    Args:
        state: Mutable cache/state dict (expects 'rows_state', 'manual_*_df').
        CLASSIFICATION: Ordered list of class names (1-based indices).
        CLASS_TITLE: Name of the classification column.
        keys: Key columns (e.g., ['plate','well','tile','mask_label']).

    Returns:
        None. Updates are written into state['manual_classified_df'] and state['manual_unclassified_df'].
    """
    selected_rows, unselected_rows = [], []
    for r in state.get("rows_state", []):
        meta = r["meta"]
        choice = r["dropdown"].value
        if choice and choice != "--select class--":
            cls_idx = CLASSIFICATION.index(choice) + 1  # 1-based
            rec = {**meta, CLASS_TITLE: cls_idx}
            if "_existing" in meta:
                rec["_existing"] = bool(meta["_existing"])
            if "_sprinkle" in meta:
                rec["_sprinkle"] = bool(meta["_sprinkle"])
            selected_rows.append(rec)
        else:
            unselected_rows.append(meta)

    base = _ensure_mc_schema(state.get("manual_classified_df"), CLASS_TITLE, keys)

    if selected_rows:
        to_add = pd.DataFrame(selected_rows)
        for col in ("_existing", "_sprinkle"):
            if col not in to_add.columns:
                to_add[col] = False
        # normalize key dtypes
        to_add["plate"] = pd.to_numeric(to_add["plate"], errors="coerce").astype(int)
        to_add["tile"] = pd.to_numeric(to_add["tile"], errors="coerce").astype(int)
        to_add["mask_label"] = pd.to_numeric(
            to_add["mask_label"], errors="coerce"
        ).astype(int)
        to_add["well"] = to_add["well"].astype(str)

        base_idx = base.set_index(keys, drop=False)
        add_idx = to_add.set_index(keys, drop=False)
        base_idx = base_idx.drop(index=add_idx.index, errors="ignore")
        combined = pd.concat(
            [base_idx, add_idx], axis=0, ignore_index=False
        ).reset_index(drop=True)
        state["manual_classified_df"] = combined

    if unselected_rows:
        unselected_df = pd.DataFrame(unselected_rows)
        state["manual_unclassified_df"] = pd.concat(
            [
                state.get("manual_unclassified_df", pd.DataFrame(columns=keys)),
                unselected_df,
            ],
            ignore_index=True,
        )


def _handle_click(
    state: dict, CLASSIFICATION: list, CLASS_TITLE: str, keys: list, on_relaunch
) -> None:
    """Handle the batch button press: collect labels and relaunch the UI.

    Args:
        state: Mutable cache/state dict.
        CLASSIFICATION: Ordered list of class names.
        CLASS_TITLE: Name of the classification column.
        keys: Key columns for identification.
        on_relaunch: Zero-arg callable to refresh the UI with next batch.

    Returns:
        None.
    """
    _collect_and_advance_random(state, CLASSIFICATION, CLASS_TITLE, keys)
    on_relaunch()


def _render_next_batch(
    state: dict,
    DISPLAY_CHANNEL: list,
    ADD_TRAINING_DATA: bool,
    keys: list,
    CLASS_TITLE: str,
    CLASSIFICATION: list,
    RELABEL_CLASSIFICATIONS: bool,
    TRAINING_DATASET_SELECTION: str,
    BATCH_SIZE: int,
    summary_df: pd.DataFrame,
    in_gate_df: pd.DataFrame,
    out_of_gate_df: pd.DataFrame,
    OUT_OF_GATE_COUNT: int,
    CHANNEL_INDICES: list,
    *,
    PHENOTYPE_OUTPUT_FP: Path,
    CHANNEL_NAMES: list,
    MODE: str,
    RESOLVED_COLORS: list,
    SCALE_BAR: int = 0,
    EXISTING_KEYS: Optional[set] = None,
    GATE_FEATURE_PRESENT: bool = False,
) -> None:
    """Render a page of mask rows and wire the "show next" button to advance batches.

    Args:
        state: Mutable cache/state dict.
        DISPLAY_CHANNEL: Labels above each channel panel.
        ADD_TRAINING_DATA: Whether the UI is collecting labels for training.
        keys: Key columns (e.g., ['plate','well','tile','mask_label']).
        CLASS_TITLE: Name of the classification column.
        CLASSIFICATION: Ordered list of class names.
        RELABEL_CLASSIFICATIONS: If True, prioritize keys already present in training data.
        TRAINING_DATASET_SELECTION: 'random' or 'top_n'.
        BATCH_SIZE: Count of rows per page.
        summary_df: Tile summary table.
        in_gate_df: In-gate pool for selection.
        out_of_gate_df: Out-of-gate pool for selection.
        OUT_OF_GATE_COUNT: Max number of out-of-threshold rows per page.
        CHANNEL_INDICES: Indices of channels to display.
        PHENOTYPE_OUTPUT_FP: Root output path.
        CHANNEL_NAMES: Channel names (for aligned stack validation).
        MODE: Normalized mode ('vacuole' or 'cell').
        RESOLVED_COLORS: List of ('gray' or label, (r,g,b)) tuples per channel.
        SCALE_BAR: Scale bar size in pixels (0 disables).
        EXISTING_KEYS: Keys present in existing training dataset (for relabeling priority).
        GATE_FEATURE_PRESENT: Whether thresholding is active (affects status text).

    Returns:
        None. Displays and updates widgets in-place.
    """
    clear_output(wait=True)

    # initialize container once
    if state.get("container") is None:
        state["status"] = widgets.HTML()
        state["button"] = widgets.Button(
            description="show_10_new_images",
            button_style="primary",
            layout=widgets.Layout(width="auto", min_width="200px"),
        )
        label_widgets = [
            widgets.HTML(
                f"<b>{ch}</b>", layout=widgets.Layout(width="200px", height="24px")
            )
            for ch in DISPLAY_CHANNEL
        ]
        label_widgets.append(
            widgets.HTML("", layout=widgets.Layout(width="200px", height="24px"))
        )
        right_labels = widgets.HBox(label_widgets)
        left_spacer = widgets.Box(layout=widgets.Layout(width="260px"))
        state["channel_header"] = widgets.HBox([left_spacer, right_labels])
        state["container"] = widgets.VBox([])

    # reset rows for this page
    state["rows_state"] = []

    total_remaining = len(in_gate_df) + len(out_of_gate_df)
    if total_remaining == 0:
        state["status"].value = "<b>All wells completed.</b>"
        close_btn = widgets.Button(
            description="Close", button_style="", layout=widgets.Layout(width="auto")
        )

        def _close(_):
            clear_output()

        close_btn.on_click(_close)
        display(widgets.VBox([state["status"], close_btn]))
        return

    EXISTING_KEYS = EXISTING_KEYS or set()

    # compute prioritized sets if relabeling active
    if ADD_TRAINING_DATA and RELABEL_CLASSIFICATIONS and len(EXISTING_KEYS) > 0:
        _in_keys = set(
            (int(r.plate), str(r.well), int(r.tile), int(r.mask_label))
            for r in in_gate_df.itertuples(index=False)
        )
        _out_keys = set(
            (int(r.plate), str(r.well), int(r.tile), int(r.mask_label))
            for r in out_of_gate_df.itertuples(index=False)
        )
        pri_in = EXISTING_KEYS.intersection(_in_keys)
        pri_out = EXISTING_KEYS.intersection(_out_keys)
    else:
        pri_in, pri_out = set(), set()

    next_batch_df, _ = select_next_batch_from_pools(
        in_pool_df=in_gate_df,
        out_pool_df=out_of_gate_df,
        selection_mode=TRAINING_DATASET_SELECTION,
        batch_size=BATCH_SIZE,
        keys=keys,
        summary_df=summary_df,
        out_randomizer=OUT_OF_GATE_COUNT,
        prioritized_in_keys=pri_in,
        prioritized_out_keys=pri_out,
    )

    state["last_batch_df"] = next_batch_df[keys].copy()

    _out_set = (
        set(
            (int(r.plate), str(r.well), int(r.tile), int(r.mask_label))
            for r in out_of_gate_df.itertuples(index=False)
        )
        if not out_of_gate_df.empty
        else set()
    )

    key_to_class = {}
    mcd = state.get("manual_classified_df")
    if mcd is not None and not mcd.empty:
        for p, w, t, m, cidx in mcd[
            ["plate", "well", "tile", "mask_label", CLASS_TITLE]
        ].itertuples(index=False, name=None):
            key_to_class[(int(p), str(w), int(t), int(m))] = int(cidx)

    rows_to_show = []
    for meta in next_batch_df.to_dict(orient="records"):
        key = (
            int(meta["plate"]),
            str(meta["well"]),
            int(meta["tile"]),
            int(meta["mask_label"]),
        )
        meta["_sprinkle"] = key in _out_set
        meta["_existing"] = key in EXISTING_KEYS
        pre = key_to_class.get(key)
        meta["_prefill_class_idx"] = (
            pre if (pre is not None and 1 <= pre <= len(CLASSIFICATION)) else None
        )
        rows_to_show.append(meta)

    row_widgets = [
        _render_row(
            meta,
            state=state,
            MODE=MODE,
            PHENOTYPE_OUTPUT_FP=PHENOTYPE_OUTPUT_FP,
            CHANNEL_NAMES=CHANNEL_NAMES,
            CHANNEL_INDICES=CHANNEL_INDICES,
            RESOLVED_COLORS=RESOLVED_COLORS,
            SCALE_BAR=SCALE_BAR,
            DISPLAY_CHANNEL=DISPLAY_CHANNEL,
            CLASSIFICATION=CLASSIFICATION,
        )
        for meta in rows_to_show
    ]

    def on_relaunch():
        in_left, out_left = remove_seen_from_pools(
            in_gate_df,
            out_of_gate_df,
            state.get("last_batch_df", pd.DataFrame(columns=keys)),
            keys=keys,
        )
        _render_next_batch(
            state=state,
            DISPLAY_CHANNEL=DISPLAY_CHANNEL,
            ADD_TRAINING_DATA=ADD_TRAINING_DATA,
            keys=keys,
            CLASS_TITLE=CLASS_TITLE,
            CLASSIFICATION=CLASSIFICATION,
            RELABEL_CLASSIFICATIONS=RELABEL_CLASSIFICATIONS,
            TRAINING_DATASET_SELECTION=TRAINING_DATASET_SELECTION,
            BATCH_SIZE=BATCH_SIZE,
            summary_df=summary_df,
            in_gate_df=in_left,
            out_of_gate_df=out_left,
            OUT_OF_GATE_COUNT=OUT_OF_GATE_COUNT,
            CHANNEL_INDICES=CHANNEL_INDICES,
            PHENOTYPE_OUTPUT_FP=PHENOTYPE_OUTPUT_FP,
            CHANNEL_NAMES=CHANNEL_NAMES,
            MODE=MODE,
            RESOLVED_COLORS=RESOLVED_COLORS,
            SCALE_BAR=SCALE_BAR,
            EXISTING_KEYS=EXISTING_KEYS,
            GATE_FEATURE_PRESENT=GATE_FEATURE_PRESENT,
        )

    state["button"].description = "show_10_new_images"
    for cb in list(state["button"]._click_handlers.callbacks):
        state["button"].on_click(cb, remove=True)
    state["button"].on_click(
        lambda _: _handle_click(state, CLASSIFICATION, CLASS_TITLE, keys, on_relaunch)
    )

    df_cls = state.get("manual_classified_df")
    if df_cls is None:
        df_cls = pd.DataFrame(columns=[CLASS_TITLE])
    df_unc = state.get("manual_unclassified_df")
    if df_unc is None:
        df_unc = pd.DataFrame()

    total_classified = len(df_cls)
    unit = "cells" if MODE == "cell" else "vacuoles"

    lines = [f"Displaying: {len(row_widgets)}"]
    if ADD_TRAINING_DATA:
        existing_count = (
            int((df_cls["_existing"] == True).sum())
            if (not df_cls.empty and "_existing" in df_cls.columns)
            else 0
        )
        lines.append(f"Existing training rows loaded: {existing_count}")
    lines.append(f"Remaining total: {total_remaining}")
    lines.append(f"In-range remaining: {len(in_gate_df)}")
    if GATE_FEATURE_PRESENT:
        lines.append(
            f"Out-of-range remaining: {len(out_of_gate_df)} (showing {OUT_OF_GATE_COUNT}/page)"
        )
    lines.append(f"Uncategorized (omitted): {len(df_unc)}")
    for i, cname in enumerate(CLASSIFICATION, start=1):
        count_i = int((df_cls[CLASS_TITLE] == i).sum()) if total_classified > 0 else 0
        pct_i = (
            int(round((count_i / total_classified) * 100))
            if total_classified > 0
            else 0
        )
        lines.append(f"{cname}: {count_i} ({pct_i}%)")
    lines.append(f"Total categorized: {total_classified} {unit}")
    state["status"].value = "<br/>".join(lines)

    state["container"].children = (
        [state["channel_header"]] + row_widgets + [state["button"], state["status"]]
    )
    display(state["container"])
