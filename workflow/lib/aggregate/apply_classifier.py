from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, Image as DisplayImage
from PIL import Image as PILImage
import tifffile
from matplotlib import colors as mcolors
from skimage import segmentation, measure



def _find_results_csv(base: Path) -> Tuple[pd.DataFrame, Path]:
    """
    Descriptions...
        Find a results CSV under `base` by checking common locations:
        - base/result.csv, base/results.csv, base/result*.csv
        - base/result/*.csv, base/results/*.csv
        Prefer a CSV containing BOTH 'model' and 'accuracy' (case-insensitive).
        Otherwise accept any CSV that has a 'model' column.

    Args:
        base (Path): Root directory to search under.

    Returns:
        Tuple[pandas.DataFrame, Path]:
            (DataFrame with normalized lowercase columns, the path to the CSV).

    Raises:
        FileNotFoundError: If no suitable CSV is found.
        ValueError: If CSVs exist but none contain a 'model' column.
    """
    base = Path(base)
    candidate_paths: List[Path] = []
    # Common direct filenames
    candidate_paths += [base / "result.csv", base / "results.csv"]
    # Any result*.csv in base
    candidate_paths += list(base.glob("result*.csv"))
    # CSVs inside 'result' or 'results' subfolders
    candidate_paths += list(base.glob("result/*.csv"))
    candidate_paths += list(base.glob("results/*.csv"))

    # Deduplicate while preserving order
    seen = set()
    ordered_candidates = []
    for p in candidate_paths:
        if p.exists() and p.is_file() and p.suffix.lower() == ".csv" and p not in seen:
            ordered_candidates.append(p)
            seen.add(p)

    best_with_acc: Optional[Tuple[pd.DataFrame, Path]] = None
    best_with_model: Optional[Tuple[pd.DataFrame, Path]] = None

    for csv_path in ordered_candidates:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue  # unreadable; skip

        # Normalize columns to lowercase for matching, keep original DF for values
        lower_map = {c: c.lower() for c in df.columns}
        df.columns = [lower_map[c] for c in df.columns]

        has_model = "model" in df.columns
        has_accuracy = "accuracy" in df.columns

        if has_model and has_accuracy and best_with_acc is None:
            best_with_acc = (df, csv_path)
        elif has_model and best_with_model is None:
            best_with_model = (df, csv_path)

    if best_with_acc:
        return best_with_acc
    if best_with_model:
        return best_with_model

    if ordered_candidates:
        raise ValueError(
            "Found CSVs but none contained a 'model' column: "
            + ", ".join(str(p) for p in ordered_candidates)
        )
    raise FileNotFoundError(
        f"No results CSV found under {base}. "
        "Expected result.csv/results.csv, result/*.csv, or results/*.csv."
    )


def display_pngs_in_plots_and_list_models(
    folder_path: str | Path,
    width: Optional[int] = None,
    sort_by: str = "name",
    reverse: bool = False,
    limit: Optional[int] = None,
    results_root: Optional[str | Path] = None,
    unique_models: bool = True,
) -> Tuple[List[Path], Optional[pd.DataFrame], List[str], Optional[Path]]:
    """
    Descriptions...
        Display all .png images from a 'plots' subfolder, and also look for a
        results CSV (commonly under 'result(s)/' or as 'result(s).csv') to print
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
    plots_dir = folder_path if folder_path.name.lower() == "plots" else folder_path / "plots"
    if not plots_dir.exists() or not plots_dir.is_dir():
        raise FileNotFoundError(f"Couldn't find a 'plots' folder at: {plots_dir}")

    # Collect PNGs (case-insensitive)
    pngs = [p for p in plots_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]

    # Sort images
    if sort_by == "mtime":
        pngs.sort(key=lambda p: p.stat().st_mtime, reverse=reverse)
    else:
        pngs.sort(key=lambda p: p.name.lower(), reverse=reverse)

    # Limit
    if limit is not None:
        pngs = pngs[:int(limit)]

    # Display images
    for img_path in pngs:
        display(DisplayImage(filename=str(img_path), width=width))

    # Locate and print models from a results CSV
    if results_root is None:
        # If we're inside 'plots', search parent; else search the provided folder
        results_root = folder_path if folder_path.name.lower() != "plots" else folder_path.parent
    results_root = Path(results_root)

    results_df: Optional[pd.DataFrame] = None
    results_csv_path: Optional[Path] = None
    printed_models: List[str] = []

    try:
        results_df, results_csv_path = _find_results_csv(results_root)
        if "model" not in results_df.columns:
            raise ValueError("Results CSV found but lacks 'model' column after normalization.")

        models_series = results_df["model"].dropna().astype(str).map(str.strip)
        if unique_models:
            # Preserve order while deduplicating
            printed_models = list(dict.fromkeys(models_series.tolist()))
        else:
            printed_models = models_series.tolist()

        print("Available models:")
        for m in printed_models:
            print(m)

    except (FileNotFoundError, ValueError) as e:
        # No hard failure for images; just report we couldn't list models
        print(f"[Info] Could not list models from results CSV: {e}")

    return pngs, results_df, printed_models, results_csv_path

def show_model_evaluation_pngs(
    CLASSIFIER_DIR_PATH: Union[str, Path],
    model_name: str,
    width: Optional[int] = 1200,
) -> List[Path]:
    """
    Display all .png images in CLASSIFIER_DIR_PATH/models/<model_name>/evaluation.

    Args:
        CLASSIFIER_DIR_PATH: Root classifier directory.
        model_name: Name of the model (folder under 'models/').
        width: Optional pixel width for each displayed image (None = original size).

    Returns:
        List of PNG file Paths that were displayed.

    Raises:
        FileNotFoundError: If the evaluation folder is missing or contains no PNGs.
    """
    # Use IPython's Image (not PIL) to avoid name collisions
    from IPython.display import display as _display, Image as _IPyImage

    eval_dir = Path(CLASSIFIER_DIR_PATH) / "models" / str(model_name) / "evaluation"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_dir}")

    pngs = sorted((p for p in eval_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
                  key=lambda p: p.name.lower())

    if not pngs:
        raise FileNotFoundError(f"No PNG files found in: {eval_dir}")

    for p in pngs:
        kwargs = {"filename": str(p)}
        if width is not None:
            kwargs["width"] = int(width)
        _display(_IPyImage(**kwargs))

    return pngs

def resolve_classifier_model_dill_path(
    CLASSIFIER_path: str | Path,
    model_name: Optional[str] = None,
) -> Path:
    """
    Descriptions...
        Build and validate the path to a model's .dill file:
        CLASSIFIER_PATH / 'model' / <model_name> / f"{model_name}_model.dill"

        If `model_name` is None, the function will search for a results CSV under
        CLASSIFIER_PATH (commonly 'result(s).csv' or files inside 'result(s)/'),
        pick the row with the highest 'accuracy' (case-insensitive), and use its
        'model' value as `model_name`.

    Args:
        CLASSIFIER_path (str | Path):
            The classifier root directory that contains 'model/' and results CSVs.
        model_name (str | None):
            The specific model name folder to use, or None to auto-select the best model
            by 'accuracy' from the results CSV.

    Returns:
        Tuple[Path, str]:
            (dill_path, model_name)

    Raises:
        FileNotFoundError:
            If the model folder doesn't exist or the dill file is missing.
        ValueError:
            If model_name is None and a suitable results CSV with 'model' and 'accuracy'
            cannot be found or parsed.
    """
    root = Path(CLASSIFIER_path)

    # If model_name not provided, pick best by accuracy from results
    if model_name is None:
        df, csv_path = _find_results_csv(root)
        if "model" not in df.columns:
            raise ValueError(f"'model' column not found in results CSV: {csv_path}")

        # Try to locate an accuracy-like column (strictly 'accuracy' per spec, but robust to variants)
        accuracy_col = None
        for cand in ["accuracy", "acc", "val_accuracy"]:
            if cand in df.columns:
                accuracy_col = cand
                break
        if accuracy_col is None:
            raise ValueError(
                f"No 'accuracy' column found in results CSV: {csv_path}. "
                "Expected a column named 'accuracy'."
            )

        # Coerce to numeric; tolerate percentages like '91.2%'
        acc_series = df[accuracy_col].astype(str).str.strip()
        acc_numeric = pd.to_numeric(acc_series.str.rstrip("%"), errors="coerce")
        if acc_numeric.isna().all():
            raise ValueError(
                f"Could not parse numeric accuracy values from column '{accuracy_col}' in {csv_path}."
            )

        best_idx = acc_numeric.idxmax()
        best_row = df.loc[best_idx]
        model_name = str(best_row["model"]).strip()

        if not model_name:
            raise ValueError(
                f"Best row by '{accuracy_col}' does not contain a valid 'model' value (CSV: {csv_path})."
            )

    # Construct the dill path
    model_dir = root / "models" / model_name
    dill_path = model_dir / f"{model_name}_model.dill"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    if not dill_path.is_file():
        raise FileNotFoundError(f"Dill file not found: {dill_path}")

    return dill_path, model_name


from pathlib import Path
from typing import Union, Iterable, Tuple, Dict, Any, Optional, List
import pandas as pd

def build_master_phenotype_df(
    plates: Union[str, int, Iterable[Union[str, int]]],
    wells: Union[str, int, Iterable[Union[str, int]]],
    name_suffix: str,
    parquet_dir: Union[str, Path],
    read_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    preview_cols: int = 10,
    preview_rows: int = 5,
    display_fn: Optional[Any] = None,  # pass IPython.display.display from the notebook
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load per-(plate, well) parquet files and concatenate into a master DataFrame.

    Args:
        plates: Plate ID or iterable of IDs (str/int).
        wells: Well ID or iterable of IDs (str/int).
        name_suffix: Filename suffix to use (e.g., 'phenotype_cp.parquet' or 'phenotype_vacuoles.parquet').
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
    # Normalize inputs to lists of strings
    def _to_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    plates = [str(p) for p in _to_list(plates)]
    wells = [str(w) for w in _to_list(wells)]
    parquet_dir = Path(parquet_dir)

    def build_filename(p, w, suffix):
        return f"P-{p}_W-{w}__{suffix}"

    read_kwargs = read_kwargs or {}

    loaded_parts: List[pd.DataFrame] = []
    found_files: List[str] = []
    missing_files: List[str] = []

    for p in plates:
        for w in wells:
            fname = build_filename(p, w, name_suffix)
            fpath = parquet_dir / fname
            if fpath.exists():
                try:
                    df_part = pd.read_parquet(fpath, **read_kwargs)
                    loaded_parts.append(df_part)
                    found_files.append(str(fpath))
                except Exception as e:
                    if verbose:
                        print(f"Failed to read {fpath}: {e}")
                    missing_files.append(f"{fpath} (read failed: {e})")
            else:
                missing_files.append(str(fpath))

    if loaded_parts:
        master_df = pd.concat(loaded_parts, ignore_index=True)
        if verbose:
            print(f"Loaded {len(found_files)} file(s), total rows: {len(master_df)}")
            print(f"Source directory: {parquet_dir}")
    else:
        master_df = pd.DataFrame()
        if verbose:
            print("No files loaded. Check 'plates_to_classify' and 'wells_to_classify' in config.")

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

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.aggregate.montage_utils import create_cell_montage, add_filenames
from lib.aggregate.eval_aggregate import summarize_cell_data

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
    verbose: bool = True,
    show_figure: bool = True,
    display_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> Tuple[Optional[plt.Figure], Optional[np.ndarray], List[np.ndarray], List[str], List[str], pd.DataFrame]:
    """
    Merge coordinates, prepare filenames, map display class names, build montages,
    and produce a summary DataFrame.

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
        raise ValueError(f"Unsupported classify_by value: {classify_by}. Use 'cell' or 'vacuole'.")

    # Validate master_phenotype_df has required columns
    required_master_cols = join_keys + [coord_cols_present[0], coord_cols_present[1]]
    missing_master = [c for c in required_master_cols if c not in master_phenotype_df.columns]
    if missing_master:
        raise KeyError("Missing required columns in master_phenotype_df: " + ", ".join(missing_master))

    # 2) Merge coordinates into classified metadata (work on a copy)
    coords_and_keys = list(required_master_cols)
    cm = classified_metadata.copy(deep=True)
    cm = cm.merge(
        master_phenotype_df[coords_and_keys].drop_duplicates(),
        on=[c for c in join_keys if c in cm.columns],
        how="left",
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
    required_cols = [c for c in ["plate", "well", "tile", coord_cols_present[0], coord_cols_present[1]] if c in cm.columns]
    before = len(cm)
    cm = cm.dropna(subset=required_cols)
    after = len(cm)
    if verbose and after < before:
        print(f"Dropped {before - after} rows with missing plate/well/tile/coords before montage.")

    for col in ["plate", "tile"]:
        if col in cm.columns:
            cm[col] = cm[col].astype(int)

    # 5) Add image paths
    cm = add_filenames(cm, root_fp)

    # 6) Build display class names (mapping numeric ids to strings)
    label_to_class = class_mapping.get("label_to_class", {}) if isinstance(class_mapping, dict) else {}

    def to_display(v):
        try:
            return label_to_class.get(int(v), v)
        except Exception:
            return label_to_class.get(v, v)

    cm["__display__"] = cm[class_title].map(to_display)

    # Ordered display names from mapping, filtered to present ones
    ordered_display_all = [label_to_class[k] for k in label_to_class] if label_to_class else list(cm["__display__"].unique())
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
    cell_class_dfs = {display_name: cm[cm["__display__"] == display_name] for display_name in ordered_classes}

    # 9) Montage generation
    title_templates = {
        True: "Lowest Confidence {cell_class} - {channel}",
        False: "Highest Confidence {cell_class} - {channel}",
    }
    conf_col = f"{class_title}_confidence"

    montages: List[np.ndarray] = []
    titles: List[str] = []

    for display_name, cell_df in cell_class_dfs.items():
        if cell_df.empty:
            if verbose:
                print(f"Skipping {display_name}: no rows available for montage.")
            continue
        for ascending in [True, False]:
            montage_dict = create_cell_montage(
                cell_data=cell_df,
                channels=channels,
                selection_params={
                    "method": "sorted",
                    "sort_by": conf_col,
                    "ascending": ascending,
                },
                coordinate_cols=list(coord_cols_present),
            )
            montage = montage_dict[montage_channel]
            montages.append(montage)
            titles.append(title_templates[ascending].format(cell_class=display_name, channel=montage_channel))

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

# apply_classifier.py

def launch_rankline_ui(
    *,
    # required data/config
    classified_metadata: pd.DataFrame,
    class_title: str,
    classify_by: str,  # 'cell'/'cells'/'cp' or 'vacuole'/'vacuoles'/'vac'
    class_mapping: Dict,  # expects {"label_to_class": {id: name, ...}} or {id: name}
    phenotype_output_fp: Union[str, Path],
    channel_names: Sequence[str],           # e.g. config["phenotype"]["channel_names"]
    display_channels: Sequence[str],        # e.g. DISPLAY_CHANNEL
    channel_colors: Optional[Sequence[str]] = None,  # e.g. CHANNEL_COLORS; can be None
    # optional filters/filename formatting
    test_plate: Optional[Iterable] = None,
    test_well: Optional[Iterable] = None,
    filename_well_pad_2: bool = False,
    # scale-bar options (choose one of px directly, or µm + pixel size)
    scale_bar_px: int = 0,          # preferred explicit px length (overrides others if >0)
    scale_bar_um: float = 0.0,      # desired µm length (used if pixel_size_um>0 and scale_bar_px==0)
    pixel_size_um: float = 0.0,     # µm per pixel (for converting scale_bar_um→px)
    # UI tuning
    minimum_difference: float = 0.01,
    thumbnail_px: int = 150,
    auto_display: bool = True,
) -> widgets.VBox:
    """
    Launches an interactive, rank-based number line UI for browsing per-class examples.

    Returns:
        container (widgets.VBox): the root widget; also displayed if auto_display=True.
    """
    # ---------- basic validations ----------
    phenotype_output_fp = Path(phenotype_output_fp)
    if not phenotype_output_fp.exists():
        raise FileNotFoundError(f"PHENOTYPE_OUTPUT_FP does not exist: {phenotype_output_fp}")

    if len(set(display_channels)) != len(display_channels):
        raise ValueError("display_channels contains duplicates.")

    missing_ch = [ch for ch in display_channels if ch not in channel_names]
    if missing_ch:
        raise ValueError(f"display_channels not found in channel_names: {missing_ch}")

    channel_indices = [channel_names.index(ch) for ch in display_channels]

    # Resolve colors per display channel
    resolved_colors: List[Tuple[str, Tuple[float, float, float]]] = []
    for i, ch in enumerate(display_channels):
        name = (channel_colors[i] if (channel_colors and i < len(channel_colors)) else None)
        resolved_colors.append(("gray", (1, 1, 1)) if name is None else ("rgb", mcolors.to_rgb(name)))

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
        "aligned_cache": {}, "mask_cache": {}, "parquet_cache": {},
        "per_class": {},  # class_id -> { df, n, lo_idx, hi_idx, mid_idx, seen_windows, gmin_conf, gmax_conf }
        "class_dropdown": None, "container": None,
        "status_html": None, "numberline_html": None, "warning_html": None,
        "buttons": {}, "grid_box": None, "last_class_id": None,
    }

    # ---------- helpers: filename formatting ----------
    def _well_for_filename(well: str) -> str:
        s = str(well).strip().upper()
        if not filename_well_pad_2:
            return s
        if len(s) >= 2 and s[0].isalpha():
            try:
                return f"{s[0]}{int(s[1:]):02d}"
            except Exception:
                return s
        return s

    # ---------- image I/O & rendering ----------
    def _robust_norm(img2d: np.ndarray) -> np.ndarray:
        img = img2d.astype(np.float32, copy=False)
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=np.nanmin(img), posinf=np.nanmax(img), neginf=np.nanmin(img))
        lo, hi = np.percentile(img, [1, 99])
        hi = max(hi, lo + 1)
        return np.clip((img - lo) / (hi - lo), 0, 1)

    def _colorize(img2d_norm: np.ndarray, color_tag_rgb) -> np.ndarray:
        tag, val = color_tag_rgb
        if tag == "gray":
            return np.stack([img2d_norm] * 3, axis=-1)
        r, g, b = val
        return np.stack([img2d_norm * r, img2d_norm * g, img2d_norm * b], axis=-1)

    def _to_png_bytes(rgb01: np.ndarray) -> bytes:
        arr = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
        im = PILImage.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def _load_aligned_stack(plate: int, well: str, tile: int) -> np.ndarray:
        key = (int(plate), str(well), int(tile))
        if key in STATE["aligned_cache"]:
            return STATE["aligned_cache"][key]
        wname = _well_for_filename(well)
        images_dir = phenotype_output_fp / "images"
        p1 = images_dir / f"P-{plate}_W-{wname}_T-{tile}__aligned.tiff"
        p2 = images_dir / f"P-{plate}_W-{wname}_T-{tile}__aligned.tif"
        path = p1 if p1.exists() else (p2 if p2.exists() else None)
        if path is None:
            raise FileNotFoundError(f"Aligned TIFF not found for P-{plate} W-{wname} T-{tile}")
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[0] != len(channel_names) and arr.shape[-1] == len(channel_names):
            arr = np.moveaxis(arr, -1, 0)
        if arr.ndim != 3:
            raise ValueError(f"Aligned TIFF must be 3D; got {arr.shape}")
        if arr.shape[0] != len(channel_names):
            raise ValueError("Channel count mismatch.")
        STATE["aligned_cache"][key] = arr
        return arr

    def _load_mask_labels(mode_: str, plate: int, well: str, tile: int) -> np.ndarray:
        key = (mode_, int(plate), str(well), int(tile))
        if key in STATE["mask_cache"]:
            return STATE["mask_cache"][key]
        wname = _well_for_filename(well)
        images_dir = phenotype_output_fp / "images"
        candidates = (
            [images_dir / f"P-{plate}_W-{wname}_T-{tile}__identified_vacuoles.tiff",
             images_dir / f"P-{plate}_W-{wname}_T-{tile}__identified_vacuoles.tif"]
            if mode_ == "vacuole" else
            [images_dir / f"P-{plate}_W-{wname}_T-{tile}__cells.tiff",
             images_dir / f"P-{plate}_W-{wname}_T-{tile}__cells.tif"]
        )
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Mask not found for mode={mode_} P-{plate} W-{wname} T-{tile}")
        labels = tifffile.imread(path)
        if labels.ndim != 2:
            raise ValueError(f"Mask must be 2D; got {labels.shape}")
        STATE["mask_cache"][key] = labels
        return labels

    def _load_parquet(mode_: str, plate: int, well: str) -> pd.DataFrame:
        key = (mode_, int(plate), str(well))
        if key in STATE["parquet_cache"]:
            return STATE["parquet_cache"][key]
        wname = _well_for_filename(well)
        pq_dir = phenotype_output_fp / "parquets"
        pq = pq_dir / (f"P-{plate}_W-{wname}__phenotype_vacuoles.parquet" if mode_ == "vacuole"
                       else f"P-{plate}_W-{wname}__phenotype_cp.parquet")
        if not pq.exists():
            raise FileNotFoundError(f"Parquet not found: {pq}")
        df = pd.read_parquet(pq)
        STATE["parquet_cache"][key] = df
        return df

    def _get_coords_for_mask(mode_: str, plate: int, well: str, tile: int, mask_label: int) -> Tuple[int, int]:
        df = _load_parquet(mode_, plate, well)
        if mode_ == "vacuole":
            sub = df[(df["tile"] == tile) & (df["vacuole_id"] == mask_label)]
            if sub.empty:
                raise KeyError("No parquet row for vacuole.")
            return int(sub.iloc[0]["vacuole_i"]), int(sub.iloc[0]["vacuole_j"])
        label_col = "cell_id" if "cell_id" in df.columns else ("label" if "label" in df.columns else None)
        if label_col is None:
            raise KeyError("Missing cell id / label.")
        sub = df[(df["tile"] == tile) & (df[label_col] == mask_label)]
        if sub.empty:
            raise KeyError("No parquet row for cell.")
        return int(sub.iloc[0]["cell_i"]), int(sub.iloc[0]["cell_j"])

    def _compute_crop_bounds(mode_: str, plate: int, well: str, tile: int, mask_label: int,
                             img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        H, W = img_shape
        labels = _load_mask_labels(mode_, plate, well, tile)
        mask = (labels == mask_label)
        if np.any(mask):
            props = measure.regionprops(mask.astype(np.uint8))
            if props:
                r = props[0]
                h = r.bbox[2] - r.bbox[0]; w = r.bbox[3] - r.bbox[1]
                half = int(np.ceil(max(h, w) / 2.0) + 6)
                half = max(20, min(half, max(H, W)))
            else:
                half = 20
        else:
            half = 20
        ci, cj = _get_coords_for_mask(mode_, plate, well, tile, mask_label)
        y0 = max(0, ci - half); y1 = min(H, ci + half)
        x0 = max(0, cj - half); x1 = min(W, cj + half)
        return y0, y1, x0, x1

    # ---------- filtering & class table ----------
    def _canon_plate(v):
        if pd.isna(v): return None
        if isinstance(v, (int, np.integer)): return str(int(v))
        if isinstance(v, float) and np.isfinite(v) and abs(v - round(v)) < 1e-8: return str(int(round(v)))
        s = str(v).strip()
        return s[:-2] if s.endswith(".0") else s

    def _canon_well(v):
        if pd.isna(v): return None
        return str(v).strip().upper()

    def _canon_list(vals, plate=False, well=False):
        if vals is None: return None
        if not isinstance(vals, (list, tuple, set)): vals = [vals]
        return [(_canon_plate(v) if plate else (_canon_well(v) if well else v)) for v in vals]

    def filter_classified_metadata(df: pd.DataFrame) -> pd.DataFrame:
        if "plate" not in df.columns or "well" not in df.columns:
            raise KeyError("Expected 'plate' and 'well' in classified_metadata.")
        df = df.copy()
        df["_plate_canon"] = df["plate"].map(_canon_plate)
        df["_well_canon"]  = df["well"].map(_canon_well)
        cplates = _canon_list(test_plate, plate=True)
        cwells  = _canon_list(test_well,  well=True)
        if cplates is None and cwells is None: return df
        if cplates is None: return df[df["_well_canon"].isin(cwells)]
        if cwells  is None: return df[df["_plate_canon"].isin(cplates)]
        return df[(df["_plate_canon"].isin(cplates)) & (df["_well_canon"].isin(cwells))]

    def _prepare_class_table(df_all: pd.DataFrame, class_id) -> pd.DataFrame:
        if class_title not in df_all.columns:
            raise KeyError(f"Missing class column '{class_title}'.")
        conf_col = f"{class_title}_confidence"
        if conf_col not in df_all.columns:
            raise KeyError(f"Missing confidence column '{conf_col}'.")
        mask = df_all[class_title].astype(str) == str(class_id)
        df = df_all.loc[mask].copy()
        df = df[df[conf_col].notna()].copy()
        df.sort_values(conf_col, ascending=True, inplace=True, kind="mergesort")
        df["__rank"] = np.arange(1, len(df) + 1, dtype=int)  # 1..N
        return df

    # ---------- scale bar ----------
    SCALE_BAR_MARGIN_FRAC = 0.02
    SCALE_BAR_THICK_FRAC  = 0.01
    SCALE_BAR_MIN_THICK   = 2
    SCALE_BAR_COLOR       = 1.0
    SCALE_BAR_DASHED_IF_TOO_LONG = True
    SCALE_BAR_DASH_COUNT  = 5

    def _get_scale_bar_px() -> int:
        if int(scale_bar_px) > 0: return int(scale_bar_px)
        if scale_bar_um > 0 and pixel_size_um > 0:
            return int(round(scale_bar_um / pixel_size_um))
        return 0

    def _overlay_scale_bar_inplace(img_rgb01: np.ndarray):
        if img_rgb01.ndim != 3 or img_rgb01.shape[2] != 3: return
        Hc, Wc = img_rgb01.shape[:2]
        bar_px = _get_scale_bar_px()
        if bar_px <= 0: return
        m  = max(2, int(round(min(Hc, Wc) * SCALE_BAR_MARGIN_FRAC)))
        th = max(SCALE_BAR_MIN_THICK, int(round(min(Hc, Wc) * SCALE_BAR_THICK_FRAC)))
        dashed = False
        if bar_px + 2*m > Wc:
            if SCALE_BAR_DASHED_IF_TOO_LONG:
                dashed = True
                bar_px = Wc - 2*m
                if bar_px <= 0: return
            else:
                bar_px = Wc - 2*m
                if bar_px <= 0: return
        y_end = Hc - m - 1
        y_start = max(0, y_end - th + 1)
        if not dashed:
            x_end = Wc - m - 1
            x_start = max(0, x_end - bar_px + 1)
            img_rgb01[y_start:y_end+1, x_start:x_end+1, :] = SCALE_BAR_COLOR
        else:
            start_x = m; end_x = Wc - m - 1
            total = max(0, end_x - start_x + 1)
            if total <= 0: return
            segs = 2 * max(1, int(SCALE_BAR_DASH_COUNT)) + 1
            group = total / segs
            dash_len = max(1, int(round(group)))
            for i in range(max(1, int(SCALE_BAR_DASH_COUNT))):
                xs = int(round(start_x + group * (2*i + 1)))
                xe = min(end_x, xs + dash_len - 1)
                if xs <= xe:
                    img_rgb01[y_start:y_end+1, xs:xe+1, :] = SCALE_BAR_COLOR

    # ---------- small utilities ----------
    def _window_indices_around(idx: int, n: int, k: int = 5) -> List[int]:
        if n <= 0: return []
        if n <= k: return list(range(n))
        left = max(0, idx - (k//2))
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
        df_all = filter_classified_metadata(classified_metadata)
        conf_col = f"{class_title}_confidence"
        df_class = _prepare_class_table(df_all, class_id)   # ASC by conf; adds __rank
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
        return st["mid_idx"] if not win else win[len(win)//2]

    # ---------- number-line & status renderers ----------
    def _render_numberline_html(class_id: int):
        st = STATE["per_class"][class_id]
        df = st["df"]; conf_col = f"{class_title}_confidence"
        n = st["n"]
        if n == 0:
            STATE["numberline_html"].value = "<i>No items to display.</i>"
            return

        lo_i, hi_i, mid_i = st["lo_idx"], st["hi_idx"], st["mid_idx"]
        lo_c = float(df.loc[lo_i, conf_col]); hi_c = float(df.loc[hi_i, conf_col]); mid_c = float(df.loc[mid_i, conf_col])

        win = _window_indices_for_class(class_id, k=5)
        left_i, right_i = (win[0], win[-1]) if win else (mid_i, mid_i)
        left_c = float(df.loc[left_i, conf_col]); right_c = float(df.loc[right_i, conf_col])

        p_wmin  = _pct_rank(lo_i,   n)
        p_wmax  = _pct_rank(hi_i,   n)
        p_mid   = _pct_rank(mid_i,  n)
        p_left  = _pct_rank(left_i, n)
        p_right = _pct_rank(right_i, n)

        width_pct = max(0.5, p_right - p_left)
        left_pct  = min(max(0.0, p_left), 100.0 - 0.5)

        def tick(left_pct, label, cls, thick=False):
            w = 3 if thick else 2
            return f"<div class='tick {cls}' style='left:{left_pct:.2f}%; border-left-width:{w}px' title='{label}'></div>"

        gmin_conf = st["gmin_conf"]; gmax_conf = st["gmax_conf"]

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
                 title='Window ranks {left_i+1}–{right_i+1} (conf {left_c:.5f}–{right_c:.5f})'></div>

            {tick(0.0,   f'Global Min (rank 1) conf {gmin_conf:.5f}', 'gmin')}
            {tick(p_wmin, f'Working Min (rank {lo_i+1}) conf {lo_c:.5f}', 'wmin', thick=True)}
            <div class='mid-tri' style='left:{p_mid:.2f}%' title='Mid (rank {mid_i+1}) conf {mid_c:.5f}'></div>
            {tick(p_wmax, f'Working Max (rank {hi_i+1}) conf {hi_c:.5f}', 'wmax', thick=True)}
            {tick(100.0, f'Global Max (rank {n}) conf {gmax_conf:.5f}', 'gmax')}
          </div>

          <div class='legend'>
            <span class='g'>Global Min: rank 1<span class='small'>(conf {gmin_conf:.5f})</span></span>
            <span class='w'>Working Min: rank {lo_i+1}<span class='small'>(conf {lo_c:.5f})</span></span>
            <span class='m'>Mid: rank {mid_i+1}<span class='small'>(conf {mid_c:.5f})</span></span>
            <span class='w'>Working Max: rank {hi_i+1}<span class='small'>(conf {hi_c:.5f})</span></span>
            <span class='g'>Global Max: rank {n}<span class='small'>(conf {gmax_conf:.5f})</span></span>
            <span class='winl'>Blue box = Current window</span>
          </div>
        </div>
        """
        STATE["numberline_html"].value = html

    def _render_status_text(class_id: int):
        st = STATE["per_class"][class_id]
        if st["n"] == 0:
            STATE["status_html"].value = "<i>No items in this class after filtering.</i>"
            return
        df = st["df"]; conf_col = f"{class_title}_confidence"
        lo_i, hi_i, mid_i = st["lo_idx"], st["hi_idx"], st["mid_idx"]
        lo_c = float(df.loc[lo_i, conf_col]); hi_c = float(df.loc[hi_i, conf_col]); mid_c = float(df.loc[mid_i, conf_col])
        STATE["status_html"].value = (
            f"Class “{_class_name_for_id(class_id)}” — "
            f"Working ranks: {lo_i+1} (min) | {mid_i+1} (mid) | {hi_i+1} (max) "
            f"&nbsp;&nbsp;[conf: {lo_c:.5f} | {mid_c:.5f} | {hi_c:.5f}]"
        )

    # ---------- rendering of one column ----------
    def _render_one_column(row: pd.Series, thumb_px: int = thumbnail_px) -> widgets.VBox:
        plate = int(row["plate"]); well = str(row["well"]); tile = int(row["tile"])
        mask_label = int(row[id_col]); conf = float(row[f"{class_title}_confidence"])
        rank = int(row["__rank"]); n_in_class = int(row["__total_in_class"])

        stack = _load_aligned_stack(plate, well, tile)
        H, W = stack.shape[1], stack.shape[2]
        labels_full = _load_mask_labels(mode, plate, well, tile)

        y0, y1, x0, x1 = _compute_crop_bounds(mode, plate, well, tile, mask_label, (H, W))
        labels_crop = labels_full[y0:y1, x0:x1]
        mask_crop = (labels_crop == mask_label)

        single_widgets: List[widgets.Image] = []
        merged = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        for ch_idx, color_tag_rgb in zip(channel_indices, resolved_colors):
            ch_crop = stack[ch_idx, y0:y1, x0:x1]
            ch_norm = _robust_norm(ch_crop)
            ch_rgb  = _colorize(ch_norm, color_tag_rgb)
            iw = widgets.Image(value=_to_png_bytes(ch_rgb), format='png')
            iw.layout = widgets.Layout(width=f'{thumb_px}px', height=f'{thumb_px}px')
            single_widgets.append(iw)
            merged += ch_rgb
        merged = np.clip(merged, 0, 1)

        if np.any(mask_crop):
            boundary = segmentation.find_boundaries(mask_crop, mode='outer')
            coords = np.argwhere(boundary)
            if len(coords) > 0:
                merged[coords[::2, 0], coords[::2, 1], :] = 1.0

        _overlay_scale_bar_inplace(merged)

        merged_w = widgets.Image(value=_to_png_bytes(merged), format='png')
        merged_w.layout = widgets.Layout(width=f'{thumb_px}px', height=f'{thumb_px}px')

        meta_line = f"P-{plate} W-{well} T-{tile} | mask {mask_label}"
        conf_line = f"Confidence: {conf:.5f}   (Rank {rank}/{n_in_class})"
        lbl = widgets.HTML(f"<div style='text-align:center; font-size:12px'>{meta_line}<br>{conf_line}</div>")
        return widgets.VBox(single_widgets + [merged_w, lbl], layout=widgets.Layout(align_items='center'))

    # ---------- UI state helpers ----------
    def _recenter_mid_within_working(class_id: int):
        st = STATE["per_class"][class_id]
        st["mid_idx"] = _idx_mid(st["lo_idx"], st["hi_idx"])

    def _set_working_bounds(class_id: int, lo_idx: int, hi_idx: int):
        st = STATE["per_class"][class_id]
        st["lo_idx"] = int(lo_idx); st["hi_idx"] = int(hi_idx)
        _recenter_mid_within_working(class_id)

    # ---------- UI update/draw ----------
    def _update_button_states(class_id: int):
        st = STATE["per_class"][class_id]; n = st["n"]
        if n == 0:
            for b in STATE["buttons"].values():
                b.disabled = True
            return
        win = _window_indices_for_class(class_id, k=5)
        left_i, right_i = win[0], win[-1]
        STATE["buttons"]["left1"].disabled  = (left_i - 1 < 0)
        STATE["buttons"]["left5"].disabled  = (left_i - 5 < 0)
        STATE["buttons"]["right1"].disabled = (right_i + 1 > n - 1)
        STATE["buttons"]["right5"].disabled = (right_i + 5 > n - 1)
        within = (left_i >= st["lo_idx"]) and (right_i <= st["hi_idx"])
        STATE["buttons"]["good"].disabled = not within
        STATE["buttons"]["bad"].disabled  = not within

    def _redraw_class(class_id: int):
        clear_output(wait=True)
        st = STATE["per_class"][class_id]
        STATE["container"].children = []

        header = widgets.HBox([
            STATE["class_dropdown"],
            widgets.HTML("<div style='width:8px'></div>"),
            STATE["buttons"]["left5"], STATE["buttons"]["left1"],
            widgets.HTML("<div style='width:8px'></div>"),
            STATE["buttons"]["right1"], STATE["buttons"]["right5"],
            widgets.HTML("<div style='width:16px'></div>"),
            STATE["buttons"]["good"],
            STATE["buttons"]["bad"],
            STATE["buttons"]["refresh"],
        ])

        _render_status_text(class_id)
        _render_numberline_html(class_id)

        df = st["df"]; n = st["n"]; warnings = []
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
            lo_c = float(df.loc[st["lo_idx"], conf_col]); hi_c = float(df.loc[st["hi_idx"], conf_col])
            if (hi_c - lo_c) < float(minimum_difference):
                warnings.append(f"<b>MINIMUM_DIFFERENCE REACHED!!! (span={hi_c - lo_c:.5f})</b>")

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
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        st["mid_idx"] = max(0, st["mid_idx"] - 1); _redraw_class(cid)

    def _on_left5_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        st["mid_idx"] = max(0, st["mid_idx"] - 5); _redraw_class(cid)

    def _on_right1_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        st["mid_idx"] = min(st["n"] - 1, st["mid_idx"] + 1); _redraw_class(cid)

    def _on_right5_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        st["mid_idx"] = min(st["n"] - 1, st["mid_idx"] + 5); _redraw_class(cid)

    def _on_good_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        center_idx = _window_mid_index(cid)
        _set_working_bounds(cid, st["lo_idx"], center_idx)
        _redraw_class(cid)

    def _on_bad_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        center_idx = _window_mid_index(cid)
        _set_working_bounds(cid, center_idx, st["hi_idx"])
        _redraw_class(cid)

    def _on_refresh_clicked(_):
        cid = STATE["last_class_id"];  st = STATE["per_class"][cid]
        if st["n"] == 0: return
        st["lo_idx"] = 0; st["hi_idx"] = st["n"] - 1; st["seen_windows"] = set()
        _recenter_mid_within_working(cid); _redraw_class(cid)

    def _on_class_changed(change):
        if change["name"] != "value": return
        cname = change["new"]
        inv = {v: k for k, v in class_mapping.get("label_to_class", class_mapping).items()}
        cid = inv.get(cname, None)
        if cid is None: return
        STATE["last_class_id"] = cid
        _per_class_init(cid)
        _redraw_class(cid)

    # ---------- boot ----------
    if STATE["container"] is None:
        label_to_class = class_mapping.get("label_to_class", class_mapping)
        class_names = [label_to_class[k] for k in sorted(label_to_class.keys(),
                        key=lambda x: int(x) if str(x).isdigit() else str(x))]
        dd = widgets.Dropdown(options=class_names, description="Class:", layout=widgets.Layout(width='280px'))
        dd.observe(_on_class_changed, names="value")
        STATE["class_dropdown"] = dd

        # nav + decisions
        btn_l5 = widgets.Button(description="<<<", layout=widgets.Layout(width="80px"))
        btn_l1 = widgets.Button(description="<",   layout=widgets.Layout(width="80px"))
        btn_r1 = widgets.Button(description=">",   layout=widgets.Layout(width="80px"))
        btn_r5 = widgets.Button(description=">>>", layout=widgets.Layout(width="80px"))
        btn_l5.on_click(_on_left5_clicked); btn_l1.on_click(_on_left1_clicked)
        btn_r1.on_click(_on_right1_clicked); btn_r5.on_click(_on_right5_clicked)

        btn_good = widgets.Button(description="good", button_style='success', layout=widgets.Layout(width="120px"))
        btn_bad  = widgets.Button(description="bad",  button_style='danger',  layout=widgets.Layout(width="120px"))
        btn_ref  = widgets.Button(description="refresh", button_style='info', layout=widgets.Layout(width="120px"))
        btn_good.on_click(_on_good_clicked); btn_bad.on_click(_on_bad_clicked); btn_ref.on_click(_on_refresh_clicked)

        STATE["buttons"] = {"left5": btn_l5, "left1": btn_l1, "right1": btn_r1, "right5": btn_r5,
                            "good": btn_good, "bad": btn_bad, "refresh": btn_ref}

        STATE["status_html"]     = widgets.HTML()
        STATE["numberline_html"] = widgets.HTML()
        STATE["warning_html"]    = widgets.HTML()
        STATE["grid_box"]        = widgets.HBox([], layout=widgets.Layout(align_items='flex-start', justify_content='space-between'))
        STATE["container"]       = widgets.VBox([])

    if STATE["last_class_id"] is None:
        first_name = STATE["class_dropdown"].options[0]
        inv = {v: k for k, v in class_mapping.get("label_to_class", class_mapping).items()}
        STATE["last_class_id"] = inv[first_name]
        _per_class_init(STATE["last_class_id"])
        _redraw_class(STATE["last_class_id"])
    else:
        cid = STATE["last_class_id"]
        _per_class_init(cid)
        _redraw_class(cid)

    return STATE["container"]
