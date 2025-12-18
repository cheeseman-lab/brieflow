"""This module provides functions for testing classifiers and determining confidence thresholds.

It includes utilities for displaying model evaluation plots, loading phenotype data,
creating class montages, and an interactive UI for selecting confidence thresholds
based on ranked predictions.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display, Image as DisplayImage
from matplotlib import colors as mcolors

from lib.aggregate.eval_aggregate import summarize_cell_data
from lib.aggregate.montage_utils import (
    add_filenames,
    create_class_confidence_montages,
)
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
from lib.shared.file_utils import get_filename, parse_filename


def display_pngs_in_plots_and_list_models(
    folder_path: str | Path,
    width: Optional[int] = None,
    sort_by: str = "name",
    reverse: bool = False,
    limit: Optional[int] = None,
    results_root: Optional[str | Path] = None,
    unique_models: bool = True,
) -> Tuple[List[Path], Optional[pd.DataFrame], List[str], Optional[Path]]:
    """Display all .png images from a 'plots' subfolder.

    Also load a results CSV at '<run_root>/results/multiclass_classifier_results.csv' to print
    out the available models listed in its 'model' column.

    Args:
        folder_path (str | Path):
            A directory that contains a 'plots/' subfolder OR is the 'plots/' folder itself.
        width (int | None):
            Optional pixel width for displayed images. If None, original size is used.
        sort_by (str):
            Sorting for images: 'name' or 'mtime'. Default: 'name'.
        reverse (bool):
            Reverse the sort order. Default: False.
        limit (int | None):
            Maximum number of images to display (after sorting). Default: None (no limit).
        results_root (str | Path | None):
            Where to search for the results CSV. Defaults to `folder_path`'s parent:
            - If `folder_path` is 'plots', search in its parent.
            - Else search in `folder_path`.
        unique_models (bool):
            If True, print unique model names in first-seen order; else print all rows.

    Returns:
        Tuple[List[Path], Optional[pandas.DataFrame], List[str], Optional[Path]]:
            (displayed_image_paths, results_df (if found), printed_models, results_csv_path)

    Raises:
        FileNotFoundError: If 'plots' cannot be located.
    """
    folder_path = Path(folder_path)
    plots_dir = (
        folder_path if folder_path.name.lower() == "plots" else folder_path / "plots"
    )
    if not plots_dir.exists() or not plots_dir.is_dir():
        raise FileNotFoundError(f"Couldn't find a 'plots' folder at: {plots_dir}")

    # Collect PNGs (case-insensitive)
    pngs = [
        p for p in plots_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    ]

    # Sort images
    if sort_by == "mtime":
        pngs.sort(key=lambda p: p.stat().st_mtime, reverse=reverse)
    else:
        pngs.sort(key=lambda p: p.name.lower(), reverse=reverse)

    # Limit
    if limit is not None:
        pngs = pngs[: int(limit)]

    # Display images
    for img_path in pngs:
        display(DisplayImage(filename=str(img_path), width=width))

    # Locate and print models from a results CSV
    if results_root is None:
        # If we're inside 'plots', search parent; else search the provided folder
        results_root = (
            folder_path if folder_path.name.lower() != "plots" else folder_path.parent
        )
    results_root = Path(results_root)

    results_df: Optional[pd.DataFrame] = None
    results_csv_path: Optional[Path] = None
    printed_models: List[str] = []

    # Directly load the expected CSV: <run_root>/results/multiclass_classifier_results.csv
    results_csv_path = results_root / "results" / "multiclass_classifier_results.csv"
    results_df = pd.read_csv(results_csv_path)
    models_series = results_df["model"].dropna().astype(str).map(str.strip)
    if unique_models:
        # Preserve order while deduplicating
        printed_models = list(dict.fromkeys(models_series.tolist()))
    else:
        printed_models = models_series.tolist()

    print("Available models:")
    for m in printed_models:
        print(m)

    return pngs, results_df, printed_models, results_csv_path


def show_model_evaluation_pngs(
    CLASSIFIER_DIR_PATH: Union[str, Path],
    model_name: str,
    width: Optional[int] = 1200,
) -> List[Path]:
    """Display all .png images in CLASSIFIER_DIR_PATH/models/<model_name>/evaluation.

    Args:
        CLASSIFIER_DIR_PATH: Root classifier directory.
        model_name: Name of the model (folder under 'models/').
        width: Optional pixel width for each displayed image (None = original size).

    Returns:
        List of PNG file Paths that were displayed.

    Raises:
        FileNotFoundError: If the evaluation folder is missing or contains no PNGs.
    """
    eval_dir = Path(CLASSIFIER_DIR_PATH) / "models" / str(model_name) / "evaluation"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_dir}")

    pngs = sorted(
        (p for p in eval_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: p.name.lower(),
    )

    if not pngs:
        raise FileNotFoundError(f"No PNG files found in: {eval_dir}")

    for p in pngs:
        kwargs = {"filename": str(p)}
        if width is not None:
            kwargs["width"] = int(width)
        display(DisplayImage(**kwargs))

    return pngs


def resolve_classifier_model_dill_path(
    CLASSIFIER_path: str | Path,
    model_name: Optional[str] = None,
) -> Path:
    """Resolve the path to a model's dill file.

        If model_name is None, selects the best model by accuracy from '<run_root>/results/multiclass_classifier_results.csv'.

    Args:
        CLASSIFIER_path (str | Path): Root classifier directory.
        model_name (str | None): Specific model name to use. If None, auto-select best by accuracy.

    Returns:
        Path to the model's dill file.

    Raises:
        FileNotFoundError: If the model folder or dill file is missing.
        ValueError: If model_name is None and no results CSV or accuracy column is found.
    """
    root = Path(CLASSIFIER_path)

    # If model_name not provided, pick best by accuracy from results
    if model_name is None:
        csv_path = root / "results" / "multiclass_classifier_results.csv"
        df = pd.read_csv(csv_path)

        # Expect 'model' and an accuracy-like column
        accuracy_col = None
        for cand in ["accuracy", "acc", "val_accuracy"]:
            if cand in df.columns:
                accuracy_col = cand
                break

        acc_series = df[accuracy_col].astype(str).str.strip()
        acc_numeric = pd.to_numeric(acc_series.str.rstrip("%"), errors="coerce")
        best_idx = acc_numeric.idxmax()
        best_row = df.loc[best_idx]
        model_name = str(best_row["model"]).strip()

    # Construct the dill path
    model_dir = root / "models" / model_name
    dill_path = model_dir / f"{model_name}_model.dill"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    if not dill_path.is_file():
        raise FileNotFoundError(f"Dill file not found: {dill_path}")

    return dill_path, model_name


# ============================= Business logic (testable) ============================= #


def canon_plate(v) -> Optional[str]:
    """Canonicalize plate IDs to strings without trailing '.0'.

    Returns:
        Canonicalized plate ID as string, or None if input is NaN.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and np.isfinite(v) and abs(v - round(v)) < 1e-8:
        return str(int(round(v)))
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def canon_well(v) -> Optional[str]:
    """Canonicalize well IDs to uppercase strings without surrounding whitespace.

    Returns:
        Canonicalized well ID as string, or None if input is NaN.
    """
    if pd.isna(v):
        return None
    return str(v).strip().upper()


def canon_list(vals, *, plate: bool = False, well: bool = False):
    """Canonicalize a list of plate or well IDs.

    Args:
        vals: Single value or iterable of values to canonicalize.
        plate: If True, canonicalize as plate IDs.
        well: If True, canonicalize as well IDs.

    Returns:
        List of canonicalized values.
    """
    if vals is None:
        return None
    if not isinstance(vals, (list, tuple, set)):
        vals = [vals]
    return [(canon_plate(v) if plate else (canon_well(v) if well else v)) for v in vals]


def filter_classified_metadata(
    df: pd.DataFrame,
    *,
    test_plate=None,
    test_well=None,
) -> pd.DataFrame:
    """Filter classified_metadata DataFrame by canonical plate and well IDs.

    Args:
        df: DataFrame containing classified metadata with 'plate' and 'well' columns.
        test_plate: Single plate ID or iterable of plate IDs to filter by.
        test_well: Single well ID or iterable of well IDs to filter by.

    Returns:
        Filtered DataFrame.
    """
    if "plate" not in df.columns or "well" not in df.columns:
        raise KeyError("Expected 'plate' and 'well' in classified_metadata.")
    out = df.copy()
    out["_plate_canon"] = out["plate"].map(canon_plate)
    out["_well_canon"] = out["well"].map(canon_well)
    cplates = canon_list(test_plate, plate=True)
    cwells = canon_list(test_well, well=True)
    if cplates is None and cwells is None:
        return out
    if cplates is None:
        return out[out["_well_canon"].isin(cwells)]
    if cwells is None:
        return out[out["_plate_canon"].isin(cplates)]
    return out[(out["_plate_canon"].isin(cplates)) & (out["_well_canon"].isin(cwells))]


def prepare_class_table(
    df_all: pd.DataFrame, class_title: str, class_id
) -> pd.DataFrame:
    """Prepare a DataFrame filtered to a specific class, sorted by confidence, with rank.

    Args:
        df_all: DataFrame containing classification results.
        class_title: Column name for the class/label IDs.
        class_id: Specific class ID to filter for.

    Returns:
        Filtered and ranked DataFrame for the specified class.

    """
    if class_title not in df_all.columns:
        raise KeyError(f"Missing class column '{class_title}'.")
    conf_col = f"{class_title}_confidence"
    if conf_col not in df_all.columns:
        raise KeyError(f"Missing confidence column '{conf_col}'.")
    mask = df_all[class_title].astype(str) == str(class_id)
    df = df_all.loc[mask].copy()
    df = df[df[conf_col].notna()].copy()
    df.sort_values(conf_col, ascending=True, inplace=True, kind="mergesort")
    df["__rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


def build_master_phenotype_df(
    plates: Union[str, int, Iterable[Union[str, int]]],
    wells: Union[str, int, Iterable[Union[str, int]]],
    mode: str,
    parquet_dir: Union[str, Path],
    read_kwargs: Optional[Dict[str, Any]] = {"engine": "pyarrow"},
    verbose: bool = True,
    preview_cols: int = 10,
    preview_rows: int = 5,
    display_fn: Optional[Any] = None,  # pass IPython.display.display from the notebook
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load per-(plate, well) parquet files and concatenate into a master DataFrame.

    Args:
        plates: Plate ID or iterable of IDs (str/int).
        wells: Well ID or iterable of IDs (str/int).
        mode: Object type to load ("cell" or "vacuole"). Determines the filename suffix automatically.
        parquet_dir: Directory containing the parquet files.
        read_kwargs: Optional kwargs forwarded to pd.read_parquet (e.g., {"engine": "pyarrow"}).
        verbose: If True, print status messages.
        preview_cols: Number of leading columns to preview (0 to disable).
        preview_rows: Number of rows to preview when previewing columns.
        display_fn: Optional display function (e.g., IPython.display.display) for notebook previews.

    Returns:
        master_df: Concatenated DataFrame of all successfully loaded files (empty if none).
        info: Dict with metadata:
              {
                "found_files": List[str],
                "missing_files": List[str],
                "files_loaded": int,
                "total_rows": int,
                "parquet_dir": str,
              }
    """

    # Determine name_suffix from mode
    if mode == "cell":
        name_suffix = "phenotype_cp.parquet"
    elif mode == "vacuole":
        name_suffix = "phenotype_vacuoles.parquet"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'cell' or 'vacuole'.")

    # Normalize inputs to lists of strings
    def _to_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    plates = [str(p) for p in _to_list(plates)]
    wells = [str(w) for w in _to_list(wells)]
    parquet_dir = Path(parquet_dir)

    # Parse name_suffix into info_type and file_type using shared utility
    # Accepts values like 'phenotype_cp.parquet' or just 'phenotype_cp'
    _, info_type, file_type = parse_filename(name_suffix)
    if not file_type:
        file_type = "parquet"

    read_kwargs = read_kwargs or {}

    loaded_parts: List[pd.DataFrame] = []
    found_files: List[str] = []
    missing_files: List[str] = []

    for p in plates:
        for w in wells:
            wnorm = well_for_filename(w)
            fname = get_filename({"plate": p, "well": wnorm}, info_type, file_type)
            fpath = parquet_dir / fname
            candidates = [fpath]
            # For phenotype_cp, also try phenotype_cp_min as a fallback
            if info_type == "phenotype_cp" and file_type.lower() == "parquet":
                alt_name = get_filename(
                    {"plate": p, "well": wnorm}, "phenotype_cp_min", file_type
                )
                candidates.append(parquet_dir / alt_name)

            chosen = next((c for c in candidates if c.exists()), None)
            if chosen is not None:
                try:
                    df_part = pd.read_parquet(chosen, **read_kwargs)
                    loaded_parts.append(df_part)
                    found_files.append(str(chosen))
                except Exception as e:
                    if verbose:
                        print(f"Failed to read {chosen}: {e}")
                    missing_files.append(f"{chosen} (read failed: {e})")
            else:
                # Report the primary expected path only to keep the list concise
                missing_files.append(str(fpath))

    if loaded_parts:
        master_df = pd.concat(loaded_parts, ignore_index=True)
        if verbose:
            print(f"Loaded {len(found_files)} file(s), total rows: {len(master_df)}")
            print(f"Source directory: {parquet_dir}")
    else:
        master_df = pd.DataFrame()
        if verbose:
            print(
                "No files loaded. Check 'plates_to_classify' and 'wells_to_classify' in config."
            )

    if missing_files and verbose:
        print("Missing files (not found):")
        for m in missing_files:
            print(" -", m)

    # Optional preview (notebook-friendly)
    if preview_cols > 0 and not master_df.empty and display_fn is not None:
        cols = list(master_df.columns[:preview_cols])
        print("Preview columns:", cols)
        display_fn(master_df.head(preview_rows)[cols])

    info = {
        "found_files": found_files,
        "missing_files": missing_files,
        "files_loaded": len(found_files),
        "total_rows": int(len(master_df)),
        "parquet_dir": str(parquet_dir),
    }
    return master_df, info


def build_montages_and_summary(
    *,
    master_phenotype_df: pd.DataFrame,
    classified_metadata: pd.DataFrame,
    classify_by: str,
    class_mapping: Dict,
    class_title: str,
    root_fp: Union[str, Path],
    channels: Sequence[str],
    montage_channel: str,
    collapse_cols: Sequence[str],
    verbose: bool = False,
    show_figure: bool = True,
    display_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> Tuple[
    Optional[plt.Figure],
    Optional[np.ndarray],
    List[np.ndarray],
    List[str],
    List[str],
    pd.DataFrame,
]:
    """Merge coordinates, prepare filenames, map display class names, build montages and produce a summary DataFrame.

    Uses imported helpers:
      - add_filenames (lib.aggregate.montage_utils)
      - create_cell_montage (lib.aggregate.montage_utils)
      - summarize_cell_data (lib.aggregate.eval_aggregate)

    Args:
        master_phenotype_df: Master DF containing coordinates and keys.
        classified_metadata: Classified rows to visualize/summarize (will NOT be mutated).
        classify_by: One of {"cell","cells","cp"} or {"vacuole","vacuoles","vac"}.
        class_mapping: Mapping dict, expected to contain "label_to_class".
        class_title: Column holding class/label IDs; confidence column is f"{class_title}_confidence".
        root_fp: Root path used by add_filenames.
        channels: Channel names passed to create_cell_montage.
        montage_channel: Which channel key to extract from the montage dict.
        collapse_cols: Columns to collapse in the summary.
        verbose: Print debug info (dtypes, missing files checks, skips).
        show_figure: If True, calls plt.show() after plotting.
        display_fn: Optional callable (e.g., IPython.display.display) to show the summary in notebooks.

    Returns:
        fig, axes, montages, titles, ordered_classes, summary_df
    """
    # 1) Determine coordinate columns and join keys
    ctype = str(classify_by).lower()
    if ctype in {"cell", "cells", "cp"}:
        coord_cols_present = ("nucleus_i", "nucleus_j")
        join_keys = ["label", "plate", "well", "tile"]
    elif ctype in {"vacuole", "vacuoles", "vac"}:
        coord_cols_present = ("vacuole_i", "vacuole_j")
        join_keys = ["vacuole_id", "plate", "well", "tile"]
    else:
        raise ValueError(
            f"Unsupported classify_by value: {classify_by}. Use 'cell' or 'vacuole'."
        )

    # Validate master_phenotype_df has required columns
    required_master_cols = join_keys + [coord_cols_present[0], coord_cols_present[1]]
    missing_master = [
        c for c in required_master_cols if c not in master_phenotype_df.columns
    ]
    if missing_master:
        raise KeyError(
            "Missing required columns in master_phenotype_df: "
            + ", ".join(missing_master)
        )

    # 2) Merge coordinates into classified metadata (work on a copy)
    coords_and_keys = list(required_master_cols)
    cm = classified_metadata.copy(deep=True)

    # Determine which join keys are available in classified_metadata
    available_join_keys = [c for c in join_keys if c in cm.columns]

    if verbose:
        print(f"Join keys needed: {join_keys}")
        print(f"Join keys available in classified_metadata: {available_join_keys}")
        print(f"Coordinate columns to add: {coord_cols_present}")

    # Check if coordinate columns are already present
    coords_already_present = all(c in cm.columns for c in coord_cols_present)

    if coords_already_present:
        if verbose:
            print("Coordinate columns already present in classified_metadata")
        # Verify join keys are also present
        missing_join_keys = [k for k in join_keys if k not in cm.columns]
        if missing_join_keys:
            raise ValueError(
                f"Coordinates present but missing required join keys: {missing_join_keys}. "
                f"This indicates malformed input data."
            )
    elif not available_join_keys:
        raise ValueError(
            f"Cannot merge coordinates: no join keys found in classified_metadata. "
            f"Required keys: {join_keys}, available columns: {list(cm.columns)}"
        )
    else:
        # Perform the merge
        cm = cm.merge(
            master_phenotype_df[coords_and_keys].drop_duplicates(),
            on=available_join_keys,
            how="left",
        )

        # Check if merge was successful
        if not all(c in cm.columns for c in coord_cols_present):
            missing_coords = [c for c in coord_cols_present if c not in cm.columns]
            raise ValueError(
                f"Merge failed to add coordinate columns {missing_coords}. "
                f"This may be due to mismatched join keys. "
                f"Used join keys: {available_join_keys}. "
                f"Check that classified_metadata has proper {available_join_keys} values."
            )

    # 3) Normalize dtypes for filename construction
    if "plate" in cm.columns:
        cm["plate"] = pd.to_numeric(cm["plate"], errors="coerce").astype("Int64")
    if "tile" in cm.columns:
        cm["tile"] = pd.to_numeric(cm["tile"], errors="coerce").astype("Int64")
    if "well" in cm.columns:
        cm["well"] = cm["well"].astype(str)
    if "label" in cm.columns and ctype in {"cell", "cells", "cp"}:
        cm["label"] = pd.to_numeric(cm["label"], errors="coerce").astype("Int64")

    # 4) Drop rows missing essentials and cast to int
    required_cols = [
        c
        for c in ["plate", "well", "tile", coord_cols_present[0], coord_cols_present[1]]
        if c in cm.columns
    ]
    before = len(cm)
    cm = cm.dropna(subset=required_cols)
    after = len(cm)
    if verbose and after < before:
        print(
            f"Dropped {before - after} rows with missing plate/well/tile/coords before montage."
        )

    for col in ["plate", "tile"]:
        if col in cm.columns:
            cm[col] = cm[col].astype(int)

    # 5) Add image paths
    cm = add_filenames(cm, root_fp)

    # 6) Build display class names (mapping numeric ids to strings)
    label_to_class = (
        class_mapping.get("label_to_class", {})
        if isinstance(class_mapping, dict)
        else {}
    )

    def to_display(v):
        try:
            return label_to_class.get(int(v), v)
        except Exception:
            return label_to_class.get(v, v)

    cm["__display__"] = cm[class_title].map(to_display)

    # Ordered display names from mapping, filtered to present ones
    ordered_display_all = (
        [label_to_class[k] for k in label_to_class]
        if label_to_class
        else list(cm["__display__"].unique())
    )
    present_display = set(cm["__display__"].unique())
    ordered_classes = [d for d in ordered_display_all if d in present_display]

    # 7) Debug prints: dtypes, sample image paths, missing file checks
    if verbose:
        cols_to_show = [c for c in ["plate", "well", "tile"] if c in cm.columns]
        print("Dtypes:")
        print(cm[cols_to_show].dtypes)
        print("Sample image paths:")
        print(cm["image_path"].head(3).to_list())
        missing_paths = [p for p in cm["image_path"].head(50) if not os.path.exists(p)]
        print(f"Missing files among first 50: {len(missing_paths)}")
        if missing_paths:
            print("Example missing:", missing_paths[:3])

    # 8) Partition rows per class (display names)
    cell_class_dfs = {
        display_name: cm[cm["__display__"] == display_name]
        for display_name in ordered_classes
    }

    # 9) Montage generation (delegate to montage_utils)
    conf_col = f"{class_title}_confidence"

    if verbose:
        print(f"\nPreparing montages...")
        print(f"cm columns: {list(cm.columns)}")
        print(f"Coordinate columns needed: {coord_cols_present}")
        coord_check = [c in cm.columns for c in coord_cols_present]
        print(f"Coordinate columns present: {coord_check}")

    montages, titles = create_class_confidence_montages(
        cell_class_dfs,
        channels,
        conf_col,
        list(coord_cols_present),
        montage_channel,
        verbose=verbose,
    )

    # 10) Plot montages in a (rows = classes, cols = 2) grid
    if len(ordered_classes) == 0:
        fig, axes = None, None
    else:
        num_rows = len(ordered_classes)
        fig, axes = plt.subplots(num_rows, 2, figsize=(10, 3 * num_rows))
        axes_arr = np.atleast_2d(axes)  # handles num_rows == 1

        for ax, title, montage in zip(axes_arr.flat, titles, montages):
            ax.imshow(montage, cmap="gray")
            ax.set_title(title, fontsize=14)
            ax.axis("off")

        if verbose:
            print("Montages of classes:")
        plt.tight_layout()
        if show_figure:
            plt.show()

    # 11) Build summary (without mutating inputs)
    cm_for_summary = cm.copy()
    cm_for_summary["class"] = cm_for_summary["__display__"]
    summary_df = summarize_cell_data(cm_for_summary, ordered_classes, collapse_cols)
    if display_fn is not None:
        display_fn(summary_df)

    # Clean up temp col in the copy before returning
    cm_for_summary.drop(columns=["__display__"], inplace=True, errors="ignore")

    return fig, axes, montages, titles, ordered_classes, summary_df


def launch_rankline_ui(
    *,
    # required data/config
    classified_metadata: pd.DataFrame,
    class_title: str,
    classify_by: str,  # 'cell'/'cells'/'cp' or 'vacuole'/'vacuoles'/'vac'
    class_mapping: Dict,  # expects {"label_to_class": {id: name, ...}} or {id: name}
    data_source: Union[str, Path],
    images_source: Optional[Union[str, Path]] = None,  # where images/masks live; defaults to data_source
    channel_names: Sequence[str],  # e.g. config["phenotype"]["channel_names"]
    display_channels: Sequence[str],  # e.g. DISPLAY_CHANNEL
    channel_colors: Optional[Sequence[str]] = None,  # e.g. CHANNEL_COLORS; can be None
    # optional filters/filename formatting
    test_plate: Optional[Iterable] = None,
    test_well: Optional[Iterable] = None,
    filename_well_pad_2: bool = False,
    # scale-bar options (choose one of px directly, or µm + pixel size)
    scale_bar_px: int = 0,  # preferred explicit px length (overrides others if >0)
    scale_bar_um: float = 0.0,  # desired µm length (used if pixel_size_um>0 and scale_bar_px==0)
    pixel_size_um: float = 0.0,  # µm per pixel (for converting scale_bar_um→px)
    # UI tuning
    minimum_difference: float = 0.01,
    thumbnail_px: int = 150,
    auto_display: bool = True,
) -> widgets.VBox:
    """Launches an interactive, rank-based number line UI for browsing per-class examples.

    Args:
        data_source: Path to data source directory (for parquets).
        images_source: Path to directory containing images/ subdirectory. If None, defaults to data_source.

    Returns:
        container (widgets.VBox): the root widget; also displayed if auto_display=True.
    """
    # ---------- basic validations ----------
    data_source = Path(data_source)
    if not data_source.exists():
        raise FileNotFoundError(
            f"data_source does not exist: {data_source}"
        )

    # Default images_source to data_source if not provided
    if images_source is None:
        images_source = data_source
    else:
        images_source = Path(images_source)
        if not images_source.exists():
            raise FileNotFoundError(
                f"images_source does not exist: {images_source}"
            )

    if len(set(display_channels)) != len(display_channels):
        raise ValueError("display_channels contains duplicates.")

    missing_ch = [ch for ch in display_channels if ch not in channel_names]
    if missing_ch:
        raise ValueError(f"display_channels not found in channel_names: {missing_ch}")

    channel_indices = [channel_names.index(ch) for ch in display_channels]

    # Resolve colors per display channel
    resolved_colors: List[Tuple[str, Tuple[float, float, float]]] = []
    for i, ch in enumerate(display_channels):
        name = (
            channel_colors[i] if (channel_colors and i < len(channel_colors)) else None
        )
        resolved_colors.append(
            ("gray", (1, 1, 1)) if name is None else ("rgb", mcolors.to_rgb(name))
        )

    # Mode & ID column
    mode = str(classify_by).lower()
    if mode == "vacuole":
        id_col = "vacuole_id"
    else:
        if "cell_id" in classified_metadata.columns:
            id_col = "cell_id"
        elif "label" in classified_metadata.columns:
            id_col = "label"
        else:
            raise KeyError("Need 'cell_id' or 'label' column for cell mode.")

    # ---------- STATE per launch ----------
    STATE = {
        "aligned_cache": {},
        "mask_cache": {},
        "parquet_cache": {},
        "per_class": {},  # class_id -> { df, n, lo_idx, hi_idx, mid_idx, seen_windows, gmin_conf, gmax_conf }
        "class_dropdown": None,
        "container": None,
        "status_html": None,
        "numberline_html": None,
        "warning_html": None,
        "buttons": {},
        "grid_box": None,
        "last_class_id": None,
    }

    # ---------- image I/O & rendering (use shared utils directly) ----------

    # ---------- filtering & class table ----------
    # use top-level business logic helpers

    # ---------- scale bar ----------
    SCALE_BAR_MARGIN_FRAC = 0.02
    SCALE_BAR_THICK_FRAC = 0.01
    SCALE_BAR_MIN_THICK = 2
    SCALE_BAR_COLOR = 1.0
    SCALE_BAR_DASHED_IF_TOO_LONG = True
    SCALE_BAR_DASH_COUNT = 5

    def _get_scale_bar_px() -> int:
        if int(scale_bar_px) > 0:
            return int(scale_bar_px)
        if scale_bar_um > 0 and pixel_size_um > 0:
            return int(round(scale_bar_um / pixel_size_um))
        return 0

    def _overlay_scale_bar_inplace(img_rgb01: np.ndarray):
        bar_px = _get_scale_bar_px()
        if bar_px <= 0:
            return
        overlay_scale_bar(
            img_rgb01,
            bar_px,
            position="bottom-right",
            color=SCALE_BAR_COLOR,
            margin_frac=SCALE_BAR_MARGIN_FRAC,
            thick_frac=SCALE_BAR_THICK_FRAC,
            min_thick=SCALE_BAR_MIN_THICK,
            dashed_if_too_long=SCALE_BAR_DASHED_IF_TOO_LONG,
            dash_count=SCALE_BAR_DASH_COUNT,
        )

    # ---------- small utilities ----------
    def _window_indices_around(idx: int, n: int, k: int = 5) -> List[int]:
        if n <= 0:
            return []
        if n <= k:
            return list(range(n))
        left = max(0, idx - (k // 2))
        right = left + k - 1
        if right >= n:
            right = n - 1
            left = right - (k - 1)
        return list(range(left, right + 1))

    def _pct_rank(idx: int, n: int) -> float:
        return 50.0 if n <= 1 else max(0.0, min(100.0, 100.0 * idx / (n - 1)))

    def _idx_mid(lo_idx: int, hi_idx: int) -> int:
        return (lo_idx + hi_idx) // 2

    def _class_name_for_id(cid) -> str:
        mapping = class_mapping.get("label_to_class", class_mapping)
        return mapping.get(cid, mapping.get(str(cid), str(cid)))

    # ---------- per-class state ----------
    def _per_class_init(class_id: int):
        if class_id in STATE["per_class"]:
            return
        df_all = filter_classified_metadata(
            classified_metadata, test_plate=test_plate, test_well=test_well
        )
        conf_col = f"{class_title}_confidence"
        df_class = prepare_class_table(
            df_all, class_title, class_id
        )  # ASC by conf; adds __rank
        n = len(df_class)
        df_class["__total_in_class"] = n
        st = {
            "df": df_class.reset_index(drop=True),
            "n": n,
            "lo_idx": 0 if n > 0 else -1,
            "hi_idx": n - 1 if n > 0 else -1,
            "mid_idx": (n - 1) // 2 if n > 0 else -1,
            "seen_windows": set(),
            "gmin_conf": float(df_class[conf_col].iloc[0]) if n > 0 else 0.0,
            "gmax_conf": float(df_class[conf_col].iloc[-1]) if n > 0 else 1.0,
        }
        STATE["per_class"][class_id] = st

    def _window_indices_for_class(class_id: int, k: int = 5) -> List[int]:
        st = STATE["per_class"][class_id]
        return _window_indices_around(st["mid_idx"], st["n"], k=k)

    def _window_mid_index(class_id: int) -> int:
        st = STATE["per_class"][class_id]
        win = _window_indices_for_class(class_id, k=5)
        return st["mid_idx"] if not win else win[len(win) // 2]

    # ---------- number-line & status renderers ----------
    def _render_numberline_html(class_id: int):
        st = STATE["per_class"][class_id]
        df = st["df"]
        conf_col = f"{class_title}_confidence"
        n = st["n"]
        if n == 0:
            STATE["numberline_html"].value = "<i>No items to display.</i>"
            return

        lo_i, hi_i, mid_i = st["lo_idx"], st["hi_idx"], st["mid_idx"]
        lo_c = float(df.loc[lo_i, conf_col])
        hi_c = float(df.loc[hi_i, conf_col])
        mid_c = float(df.loc[mid_i, conf_col])

        win = _window_indices_for_class(class_id, k=5)
        left_i, right_i = (win[0], win[-1]) if win else (mid_i, mid_i)
        left_c = float(df.loc[left_i, conf_col])
        right_c = float(df.loc[right_i, conf_col])

        p_wmin = _pct_rank(lo_i, n)
        p_wmax = _pct_rank(hi_i, n)
        p_mid = _pct_rank(mid_i, n)
        p_left = _pct_rank(left_i, n)
        p_right = _pct_rank(right_i, n)

        width_pct = max(0.5, p_right - p_left)
        left_pct = min(max(0.0, p_left), 100.0 - 0.5)

        def tick(left_pct, label, cls, thick=False):
            w = 3 if thick else 2
            return f"<div class='tick {cls}' style='left:{left_pct:.2f}%; border-left-width:{w}px' title='{label}'></div>"

        gmin_conf = st["gmin_conf"]
        gmax_conf = st["gmax_conf"]

        html = f"""
        <style>
          .nl-wrap {{ position: relative; }}
          .nl-bar {{ position: relative; height: 12px; background: #f2f2f2;
                     border-radius: 6px; margin: 8px 0 6px 0; }}
          .tick {{ position: absolute; top: -6px; height: 24px; border-left: 2px solid #777; z-index: 3; }}
          .gmin, .gmax {{ border-color:#9e9e9e; }}
          .wmin, .wmax {{ border-color:#2e7d32; }}
          .mid-tri {{ position: absolute; z-index: 4; width: 0; height: 0; top: -14px; transform: translateX(-50%);
                      border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid #000; }}
          .win {{ position:absolute; top:1px; height:10px; background: rgba(33,150,243,0.35);
                  border: 1px solid rgba(33,150,243,0.85); border-radius: 5px; z-index: 2; min-width: 6px; }}
          .legend {{ font-size: 12px; display:flex; gap:12px; margin-top:4px; flex-wrap:wrap; color:#444; }}
          .legend span::before {{ content:''; display:inline-block; width:10px; height:0; border-top:3px solid;
                                  margin-right:6px; vertical-align:middle; }}
          .legend .g::before {{ border-color:#9e9e9e; }}
          .legend .w::before {{ border-color:#2e7d32; }}
          .legend .m::before {{ border-color:#000; }}
          .legend .winl::before {{ border-color:rgba(33,150,243,0.85); }}
          .small {{ color:#666; margin-left:6px; }}
        </style>

        <div class='nl-wrap'>
          <div class='nl-bar'>
            <div class='win' style='left:{left_pct:.2f}%; width:{width_pct:.2f}%;'
                 title='Window ranks {left_i + 1}–{right_i + 1} (conf {left_c:.5f}–{right_c:.5f})'></div>

            {tick(0.0, f"Global Min (rank 1) conf {gmin_conf:.5f}", "gmin")}
            {tick(p_wmin, f"Working Min (rank {lo_i + 1}) conf {lo_c:.5f}", "wmin", thick=True)}
            <div class='mid-tri' style='left:{p_mid:.2f}%' title='Mid (rank {mid_i + 1}) conf {mid_c:.5f}'></div>
            {tick(p_wmax, f"Working Max (rank {hi_i + 1}) conf {hi_c:.5f}", "wmax", thick=True)}
            {tick(100.0, f"Global Max (rank {n}) conf {gmax_conf:.5f}", "gmax")}
          </div>

          <div class='legend'>
            <span class='g'>Global Min: rank 1<span class='small'>(conf {gmin_conf:.5f})</span></span>
            <span class='w'>Working Min: rank {lo_i + 1}<span class='small'>(conf {lo_c:.5f})</span></span>
            <span class='m'>Mid: rank {mid_i + 1}<span class='small'>(conf {mid_c:.5f})</span></span>
            <span class='w'>Working Max: rank {hi_i + 1}<span class='small'>(conf {hi_c:.5f})</span></span>
            <span class='g'>Global Max: rank {n}<span class='small'>(conf {gmax_conf:.5f})</span></span>
            <span class='winl'>Blue box = Current window</span>
          </div>
        </div>
        """
        STATE["numberline_html"].value = html

    def _render_status_text(class_id: int):
        st = STATE["per_class"][class_id]
        if st["n"] == 0:
            STATE[
                "status_html"
            ].value = "<i>No items in this class after filtering.</i>"
            return
        df = st["df"]
        conf_col = f"{class_title}_confidence"
        lo_i, hi_i, mid_i = st["lo_idx"], st["hi_idx"], st["mid_idx"]
        lo_c = float(df.loc[lo_i, conf_col])
        hi_c = float(df.loc[hi_i, conf_col])
        mid_c = float(df.loc[mid_i, conf_col])
        STATE["status_html"].value = (
            f"Class “{_class_name_for_id(class_id)}” — "
            f"Working ranks: {lo_i + 1} (min) | {mid_i + 1} (mid) | {hi_i + 1} (max) "
            f"&nbsp;&nbsp;[conf: {lo_c:.5f} | {mid_c:.5f} | {hi_c:.5f}]"
        )

    # ---------- rendering of one column ----------
    def _render_one_column(
        row: pd.Series, thumb_px: int = thumbnail_px
    ) -> widgets.VBox:
        plate = int(row["plate"])
        well = str(row["well"])
        tile = int(row["tile"])
        mask_label = int(row[id_col])
        conf = float(row[f"{class_title}_confidence"])
        rank = int(row["__rank"])
        n_in_class = int(row["__total_in_class"])

        stack = load_aligned_stack(
            images_source,
            channel_names,
            int(plate),
            str(well),
            int(tile),
            cache=STATE.get("aligned_cache"),
        )
        H, W = stack.shape[1], stack.shape[2]
        labels_full = load_mask_labels(
            images_source,
            mode,
            int(plate),
            str(well),
            int(tile),
            cache=STATE.get("mask_cache"),
        )

        y0, y1, x0, x1 = compute_crop_bounds(
            images_source,
            mode,
            int(plate),
            str(well),
            int(tile),
            int(mask_label),
            (H, W),
            mask_cache=STATE.get("mask_cache"),
            parquet_cache=STATE.get("parquet_cache"),
        )
        labels_crop = labels_full[y0:y1, x0:x1]
        mask_crop = labels_crop == mask_label

        single_widgets: List[widgets.Image] = []
        imgs, merged = compose_rgb_crops(
            stack, y0, y1, x0, x1, channel_indices, resolved_colors
        )
        for ch_rgb in imgs:
            iw = widgets.Image(value=to_png_bytes(ch_rgb), format="png")
            iw.layout = widgets.Layout(width=f"{thumb_px}px", height=f"{thumb_px}px")
            single_widgets.append(iw)

        if np.any(mask_crop):
            overlay_mask_boundary_inplace(merged, mask_crop, step=2, value=1.0)

        _overlay_scale_bar_inplace(merged)

        merged_w = widgets.Image(value=to_png_bytes(merged), format="png")
        merged_w.layout = widgets.Layout(width=f"{thumb_px}px", height=f"{thumb_px}px")

        meta_line = f"P-{plate} W-{well} T-{tile} | mask {mask_label}"
        conf_line = f"Confidence: {conf:.5f}   (Rank {rank}/{n_in_class})"
        lbl = widgets.HTML(
            f"<div style='text-align:center; font-size:12px'>{meta_line}<br>{conf_line}</div>"
        )
        return widgets.VBox(
            single_widgets + [merged_w, lbl],
            layout=widgets.Layout(align_items="center"),
        )

    # ---------- UI state helpers ----------
    def _recenter_mid_within_working(class_id: int):
        st = STATE["per_class"][class_id]
        st["mid_idx"] = _idx_mid(st["lo_idx"], st["hi_idx"])

    def _set_working_bounds(class_id: int, lo_idx: int, hi_idx: int):
        st = STATE["per_class"][class_id]
        st["lo_idx"] = int(lo_idx)
        st["hi_idx"] = int(hi_idx)
        _recenter_mid_within_working(class_id)

    # ---------- UI update/draw ----------
    def _update_button_states(class_id: int):
        st = STATE["per_class"][class_id]
        n = st["n"]
        if n == 0:
            for b in STATE["buttons"].values():
                b.disabled = True
            return
        win = _window_indices_for_class(class_id, k=5)
        left_i, right_i = win[0], win[-1]
        STATE["buttons"]["left1"].disabled = left_i - 1 < 0
        STATE["buttons"]["left5"].disabled = left_i - 5 < 0
        STATE["buttons"]["right1"].disabled = right_i + 1 > n - 1
        STATE["buttons"]["right5"].disabled = right_i + 5 > n - 1
        within = (left_i >= st["lo_idx"]) and (right_i <= st["hi_idx"])
        STATE["buttons"]["good"].disabled = not within
        STATE["buttons"]["bad"].disabled = not within

    def _redraw_class(class_id: int):
        clear_output(wait=True)
        st = STATE["per_class"][class_id]
        STATE["container"].children = []

        header = widgets.HBox(
            [
                STATE["class_dropdown"],
                widgets.HTML("<div style='width:8px'></div>"),
                STATE["buttons"]["left5"],
                STATE["buttons"]["left1"],
                widgets.HTML("<div style='width:8px'></div>"),
                STATE["buttons"]["right1"],
                STATE["buttons"]["right5"],
                widgets.HTML("<div style='width:16px'></div>"),
                STATE["buttons"]["good"],
                STATE["buttons"]["bad"],
                STATE["buttons"]["refresh"],
            ]
        )

        _render_status_text(class_id)
        _render_numberline_html(class_id)

        df = st["df"]
        n = st["n"]
        warnings = []
        if n == 0:
            STATE["warning_html"].value = "<i>No items in this class.</i>"
            STATE["grid_box"].children = []
        else:
            win = _window_indices_for_class(class_id, k=5)
            tup = tuple(win)
            if tup in st["seen_windows"]:
                warnings.append("<b>IMAGES FROM THIS WINDOW HAVE BEEN VIEWED!!!</b>")
            st["seen_windows"].add(tup)

            STATE["grid_box"].children = [_render_one_column(df.loc[i]) for i in win]

            if st["hi_idx"] - st["lo_idx"] + 1 <= 5:
                warnings.append("<b>halving limit reached</b>")

            conf_col = f"{class_title}_confidence"
            lo_c = float(df.loc[st["lo_idx"], conf_col])
            hi_c = float(df.loc[st["hi_idx"], conf_col])
            if (hi_c - lo_c) < float(minimum_difference):
                warnings.append(
                    f"<b>MINIMUM_DIFFERENCE REACHED!!! (span={hi_c - lo_c:.5f})</b>"
                )

            STATE["warning_html"].value = "<br>".join(warnings)

        _update_button_states(class_id)

        STATE["container"].children = [
            header,
            STATE["status_html"],
            STATE["numberline_html"],
            STATE["warning_html"],
            STATE["grid_box"],
        ]
        if auto_display:
            display(STATE["container"])

    # ---------- event handlers ----------
    def _on_left1_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        st["mid_idx"] = max(0, st["mid_idx"] - 1)
        _redraw_class(cid)

    def _on_left5_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        st["mid_idx"] = max(0, st["mid_idx"] - 5)
        _redraw_class(cid)

    def _on_right1_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        st["mid_idx"] = min(st["n"] - 1, st["mid_idx"] + 1)
        _redraw_class(cid)

    def _on_right5_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        st["mid_idx"] = min(st["n"] - 1, st["mid_idx"] + 5)
        _redraw_class(cid)

    def _on_good_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        center_idx = _window_mid_index(cid)
        _set_working_bounds(cid, st["lo_idx"], center_idx)
        _redraw_class(cid)

    def _on_bad_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        center_idx = _window_mid_index(cid)
        _set_working_bounds(cid, center_idx, st["hi_idx"])
        _redraw_class(cid)

    def _on_refresh_clicked(_):
        cid = STATE["last_class_id"]
        st = STATE["per_class"][cid]
        if st["n"] == 0:
            return
        st["lo_idx"] = 0
        st["hi_idx"] = st["n"] - 1
        st["seen_windows"] = set()
        _recenter_mid_within_working(cid)
        _redraw_class(cid)

    def _on_class_changed(change):
        if change["name"] != "value":
            return
        cname = change["new"]
        inv = {
            v: k for k, v in class_mapping.get("label_to_class", class_mapping).items()
        }
        cid = inv.get(cname, None)
        if cid is None:
            return
        STATE["last_class_id"] = cid
        _per_class_init(cid)
        _redraw_class(cid)

    # ---------- boot ----------
    if STATE["container"] is None:
        label_to_class = class_mapping.get("label_to_class", class_mapping)
        class_names = [
            label_to_class[k]
            for k in sorted(
                label_to_class.keys(),
                key=lambda x: int(x) if str(x).isdigit() else str(x),
            )
        ]
        dd = widgets.Dropdown(
            options=class_names,
            description="Class:",
            layout=widgets.Layout(width="280px"),
        )
        dd.observe(_on_class_changed, names="value")
        STATE["class_dropdown"] = dd

        # nav + decisions
        btn_l5 = widgets.Button(description="<<<", layout=widgets.Layout(width="80px"))
        btn_l1 = widgets.Button(description="<", layout=widgets.Layout(width="80px"))
        btn_r1 = widgets.Button(description=">", layout=widgets.Layout(width="80px"))
        btn_r5 = widgets.Button(description=">>>", layout=widgets.Layout(width="80px"))
        btn_l5.on_click(_on_left5_clicked)
        btn_l1.on_click(_on_left1_clicked)
        btn_r1.on_click(_on_right1_clicked)
        btn_r5.on_click(_on_right5_clicked)

        btn_good = widgets.Button(
            description="good",
            button_style="success",
            layout=widgets.Layout(width="120px"),
        )
        btn_bad = widgets.Button(
            description="bad",
            button_style="danger",
            layout=widgets.Layout(width="120px"),
        )
        btn_ref = widgets.Button(
            description="refresh",
            button_style="info",
            layout=widgets.Layout(width="120px"),
        )
        btn_good.on_click(_on_good_clicked)
        btn_bad.on_click(_on_bad_clicked)
        btn_ref.on_click(_on_refresh_clicked)

        STATE["buttons"] = {
            "left5": btn_l5,
            "left1": btn_l1,
            "right1": btn_r1,
            "right5": btn_r5,
            "good": btn_good,
            "bad": btn_bad,
            "refresh": btn_ref,
        }

        STATE["status_html"] = widgets.HTML()
        STATE["numberline_html"] = widgets.HTML()
        STATE["warning_html"] = widgets.HTML()
        STATE["grid_box"] = widgets.HBox(
            [],
            layout=widgets.Layout(
                align_items="flex-start", justify_content="space-between"
            ),
        )
        STATE["container"] = widgets.VBox([])

    if STATE["last_class_id"] is None:
        first_name = STATE["class_dropdown"].options[0]
        inv = {
            v: k for k, v in class_mapping.get("label_to_class", class_mapping).items()
        }
        STATE["last_class_id"] = inv[first_name]
        _per_class_init(STATE["last_class_id"])
        _redraw_class(STATE["last_class_id"])
    else:
        cid = STATE["last_class_id"]
        _per_class_init(cid)
        _redraw_class(cid)

    return STATE["container"]
