"""This module provides shared imaging and IO utilities for classifier modules.

Includes utilities for:
- Image normalization, colorization, and PNG conversion
- Aligned image stack and mask loading with optional caches
- Phenotype parquet loading with optional caches
- Mask coordinate lookup and crop bounds computation
- Scale bar and mask boundary overlays
- Helper function for finding classifier run directories
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tifffile
from PIL import Image as PILImage
from skimage import measure, segmentation

from lib.shared.file_utils import get_filename


def robust_norm(
    img2d: np.ndarray, low: float = 1.0, high: float = 99.0, eps: float = 1e-6
) -> np.ndarray:
    """Robust percentile normalization to [0,1] on a 2D array.

    Handles NaN/Inf by mapping to finite min/max first.
    """
    img = img2d.astype(np.float32, copy=False)
    if not np.isfinite(img).all():
        # Replace NaNs/Infs conservatively
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return np.zeros_like(img)
        vmin, vmax = float(finite.min()), float(finite.max())
        img = np.nan_to_num(img, nan=vmin, posinf=vmax, neginf=vmin)
    lo, hi = np.percentile(img, [low, high])
    if not np.isfinite(lo):
        lo = float(np.nanmin(img))
    if not np.isfinite(hi):
        hi = float(np.nanmax(img))
    if hi <= lo:
        hi = lo + eps
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def colorize(
    img2d_norm: np.ndarray, color_tag_rgb: Tuple[str, Tuple[float, float, float]]
) -> np.ndarray:
    """Map a normalized 2D image to RGB using either gray or a provided RGB tuple.

    color_tag_rgb format: ("gray"|"rgb", (r,g,b)); when tag is "gray" the rgb tuple is ignored.
    """
    tag, val = color_tag_rgb
    if tag == "gray":
        return np.stack([img2d_norm] * 3, axis=-1)
    r, g, b = val
    return np.stack([img2d_norm * r, img2d_norm * g, img2d_norm * b], axis=-1)


def to_png_bytes(rgb01: np.ndarray) -> bytes:
    """Convert an RGB float [0,1] array to PNG bytes."""
    arr = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    im = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def compose_rgb_crops(
    stack: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    channel_indices: Sequence[int],
    color_specs: Sequence[Tuple[str, Tuple[float, float, float]]],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compose per-channel RGB crops and a merged RGB image.

    Returns (imgs, merged) where imgs is a list of RGB float arrays per channel and
    merged is their sum clipped to [0,1].
    """
    imgs: List[np.ndarray] = []
    merged = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    for ch_idx, color_tag_rgb in zip(channel_indices, color_specs):
        ch_crop = stack[ch_idx, y0:y1, x0:x1]
        ch_norm = robust_norm(ch_crop)
        ch_rgb = colorize(ch_norm, color_tag_rgb)
        imgs.append(ch_rgb)
        merged += ch_rgb
    merged = np.clip(merged, 0.0, 1.0)
    return imgs, merged


def overlay_mask_boundary_inplace(
    img_rgb01: np.ndarray, mask_bool: np.ndarray, *, step: int = 2, value: float = 1.0
) -> None:
    """Overlay a boundary from a boolean mask onto an RGB float image in-place.

    Args:
        img_rgb01: RGB float image in [0,1].
        mask_bool: Boolean mask of the same HxW as the image crop.
        step: Subsample factor for boundary coordinates for speed.
        value: Float value to assign to boundary pixels across all channels.
    """
    if img_rgb01.ndim != 3 or img_rgb01.shape[2] != 3:
        return
    if mask_bool.size == 0 or not np.any(mask_bool):
        return
    boundary = segmentation.find_boundaries(mask_bool, mode="outer")
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return
    coords_sel = coords[:: max(1, int(step))]
    img_rgb01[coords_sel[:, 0], coords_sel[:, 1], :] = value


def well_for_filename(well: Union[str, int]) -> str:
    """Normalize well id for filenames using unpadded columns.

    Examples:
      - 'A1' -> 'A1'
      - 'A01' -> 'A1'  (strip leading zeros)
      - 'b12' -> 'B12'

    If the pattern doesn't match Letter+digits, returns uppercased string as-is.
    """
    s = str(well).strip().upper()
    m = re.match(r"^([A-Z])0*(\d{1,2})$", s)
    if not m:
        return s
    row, col = m.group(1), m.group(2)
    # int() removes any leading zeros
    return f"{row}{int(col)}"


def load_aligned_stack(
    phenotype_output_fp: Union[str, Path],
    channel_names: Sequence[str],
    plate: int,
    well: Union[str, int],
    tile: int,
    *,
    cache: Optional[Dict[Any, Any]] = None,
) -> np.ndarray:
    """Load aligned TIFF stack as (C,H,W) with channels matching channel_names."""
    phenotype_output_fp = Path(phenotype_output_fp)
    key = (int(plate), str(well), int(tile))
    if cache is not None and key in cache:
        return cache[key]

    wname = well_for_filename(well)
    # Backward-compat: also try zero-padded well variant if files were written that way
    m = re.match(r"^([A-Z])(\d{1,2})$", wname)
    wpad = f"{m.group(1)}{int(m.group(2)):02d}" if m else wname
    images_dir = phenotype_output_fp / "images"
    # Prefer using shared filename builder; try both tiff and tif
    candidates = [
        images_dir
        / get_filename(
            {"plate": plate, "well": wname, "tile": tile}, "aligned", "tiff"
        ),
        images_dir
        / get_filename({"plate": plate, "well": wname, "tile": tile}, "aligned", "tif"),
        images_dir
        / get_filename({"plate": plate, "well": wpad, "tile": tile}, "aligned", "tiff"),
        images_dir
        / get_filename({"plate": plate, "well": wpad, "tile": tile}, "aligned", "tif"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Aligned TIFF not found for P-{plate} W-{wname} T-{tile}"
        )
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif (
        arr.ndim == 3
        and arr.shape[0] != len(channel_names)
        and arr.shape[-1] == len(channel_names)
    ):
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim != 3:
        raise ValueError(f"Aligned TIFF must be 3D; got {arr.shape}")
    if arr.shape[0] != len(channel_names):
        raise ValueError("Channel count mismatch.")
    if cache is not None:
        cache[key] = arr
    return arr


def load_mask_labels(
    phenotype_output_fp: Union[str, Path],
    mode: str,
    plate: int,
    well: Union[str, int],
    tile: int,
    *,
    cache: Optional[Dict[Any, Any]] = None,
) -> np.ndarray:
    """Load 2D mask labels image for either 'vacuole' or 'cell' modes."""
    phenotype_output_fp = Path(phenotype_output_fp)
    mode_ = str(mode).lower()
    key = (mode_, int(plate), str(well), int(tile))
    if cache is not None and key in cache:
        return cache[key]

    wname = well_for_filename(well)
    # Backward-compat: also try zero-padded well variant
    m = re.match(r"^([A-Z])(\d{1,2})$", wname)
    wpad = f"{m.group(1)}{int(m.group(2)):02d}" if m else wname
    images_dir = phenotype_output_fp / "images"
    if mode_ == "vacuole":
        candidates = [
            images_dir
            / get_filename(
                {"plate": plate, "well": wname, "tile": tile},
                "identified_vacuoles",
                "tiff",
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wname, "tile": tile},
                "identified_vacuoles",
                "tif",
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wpad, "tile": tile},
                "identified_vacuoles",
                "tiff",
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wpad, "tile": tile},
                "identified_vacuoles",
                "tif",
            ),
        ]
    else:
        candidates = [
            images_dir
            / get_filename(
                {"plate": plate, "well": wname, "tile": tile}, "cells", "tiff"
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wname, "tile": tile}, "cells", "tif"
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wpad, "tile": tile}, "cells", "tiff"
            ),
            images_dir
            / get_filename(
                {"plate": plate, "well": wpad, "tile": tile}, "cells", "tif"
            ),
        ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Mask image not found for mode={mode_}, plate={plate}, well={wname}, tile={tile}."
        )
    labels = tifffile.imread(path)
    if labels.ndim != 2:
        raise ValueError(f"Mask must be 2D; got {labels.shape}")
    if cache is not None:
        cache[key] = labels
    return labels


def load_parquet(
    phenotype_output_fp: Union[str, Path],
    mode: str,
    plate: int,
    well: Union[str, int],
    *,
    cache: Optional[Dict[Any, Any]] = None,
) -> pd.DataFrame:
    """Load phenotype parquet for a specific plate/well and mode ('vacuole' or 'cell').

    For cell mode, if 'phenotype_cp.parquet' is missing, will fall back to 'phenotype_cp_min.parquet'.
    """
    phenotype_output_fp = Path(phenotype_output_fp)
    mode_ = str(mode).lower()
    key = (mode_, int(plate), well_for_filename(well))
    if cache is not None and key in cache:
        return cache[key]

    pq_dir = phenotype_output_fp / "parquets"
    wname = well_for_filename(well)
    m = re.match(r"^([A-Z])(\d{1,2})$", wname)
    wpad = f"{m.group(1)}{int(m.group(2)):02d}" if m else wname
    if mode_ == "vacuole":
        candidates = [
            pq_dir
            / get_filename(
                {"plate": plate, "well": wname}, "phenotype_vacuoles", "parquet"
            ),
            pq_dir
            / get_filename(
                {"plate": plate, "well": wpad}, "phenotype_vacuoles", "parquet"
            ),
        ]
    else:
        candidates = [
            pq_dir
            / get_filename({"plate": plate, "well": wname}, "phenotype_cp", "parquet"),
            pq_dir
            / get_filename(
                {"plate": plate, "well": wname}, "phenotype_cp_min", "parquet"
            ),
            pq_dir
            / get_filename({"plate": plate, "well": wpad}, "phenotype_cp", "parquet"),
            pq_dir
            / get_filename(
                {"plate": plate, "well": wpad}, "phenotype_cp_min", "parquet"
            ),
        ]
    pq = next((p for p in candidates if p.exists()), None)
    if pq is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Parquet not found for mode={mode_}, plate={plate}, well={wname}. Tried: {tried}"
        )
    df = pd.read_parquet(pq)
    if cache is not None:
        cache[key] = df
    return df


def get_coords_for_mask(
    phenotype_output_fp: Union[str, Path],
    mode: str,
    plate: int,
    well: Union[str, int],
    tile: int,
    mask_label: int,
    *,
    parquet_cache: Optional[Dict[Any, Any]] = None,
) -> Tuple[int, int]:
    """Retrieve (i,j) coordinates for a mask centroid from the phenotype parquet."""
    df = load_parquet(phenotype_output_fp, mode, plate, well, cache=parquet_cache)
    mode_ = str(mode).lower()
    if mode_ == "vacuole":
        sub = df[(df["tile"] == tile) & (df["vacuole_id"] == mask_label)]
        if sub.empty:
            raise KeyError(
                f"No parquet row for vacuole: P-{plate} W-{well_for_filename(well)} T-{tile} vacuole_id={mask_label}"
            )
        return int(sub.iloc[0]["vacuole_i"]), int(sub.iloc[0]["vacuole_j"])

    # Cell/cp mode: support multiple possible label columns
    label_col = None
    for cand in ("cell_id", "label", "labels"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(
            "Neither 'cell_id', 'label' nor 'labels' found in parquet columns"
        )
    sub = df[(df["tile"] == tile) & (df[label_col] == mask_label)]
    if sub.empty:
        raise KeyError(
            f"No parquet row for cell: P-{plate} W-{well_for_filename(well)} T-{tile} {label_col}={mask_label}"
        )
    return int(sub.iloc[0]["cell_i"]), int(sub.iloc[0]["cell_j"])


def compute_crop_bounds(
    phenotype_output_fp: Union[str, Path],
    mode: str,
    plate: int,
    well: Union[str, int],
    tile: int,
    mask_label: int,
    img_shape: Tuple[int, int],
    *,
    mask_cache: Optional[Dict[Any, Any]] = None,
    parquet_cache: Optional[Dict[Any, Any]] = None,
    min_half: int = 20,
    pad: int = 6,
) -> Tuple[int, int, int, int]:
    """Compute a square crop around a mask with a small margin.

    Falls back to default min_half when mask regionprops are unavailable or mask missing.
    """
    H, W = img_shape
    labels = load_mask_labels(
        phenotype_output_fp, mode, plate, well, tile, cache=mask_cache
    )
    mask = labels == mask_label
    if np.any(mask):
        props = measure.regionprops(mask.astype(np.uint8))
        if props:
            r = props[0]
            h = r.bbox[2] - r.bbox[0]
            w = r.bbox[3] - r.bbox[1]
            half = int(np.ceil(max(h, w) / 2.0) + pad)
            half = max(min_half, min(half, max(H, W)))
        else:
            half = min_half
    else:
        half = min_half

    ci, cj = get_coords_for_mask(
        phenotype_output_fp,
        mode,
        plate,
        well,
        tile,
        mask_label,
        parquet_cache=parquet_cache,
    )
    y0 = max(0, ci - half)
    y1 = min(H, ci + half)
    x0 = max(0, cj - half)
    x1 = min(W, cj + half)
    return y0, y1, x0, x1


def overlay_scale_bar(
    img_rgb01: np.ndarray,
    bar_len_px: int,
    *,
    position: str = "bottom-right",
    color: Union[float, Tuple[float, float, float]] = 1.0,
    margin_frac: float = 0.02,
    thick_frac: float = 0.01,
    min_thick: int = 2,
    dashed_if_too_long: bool = True,
    dash_count: int = 5,
) -> np.ndarray:
    """Overlay a scale bar onto an RGB float image. Returns modified image (same array if possible)."""
    if img_rgb01.ndim != 3 or img_rgb01.shape[2] != 3:
        return img_rgb01
    Hc, Wc = img_rgb01.shape[:2]
    if bar_len_px <= 0:
        return img_rgb01
    m = max(2, int(round(min(Hc, Wc) * margin_frac)))
    th = max(min_thick, int(round(min(Hc, Wc) * thick_frac)))

    # Only bottom-right is supported currently; can be extended later.
    y_end = Hc - m - 1
    y_start = max(0, y_end - th + 1)

    # Resolve color (float -> gray, tuple -> rgb multipliers)
    if isinstance(color, (int, float)):
        col = (float(color),) * 3
    else:
        col = tuple(map(float, color))

    # Clamp bar length if needed
    dashed = False
    bar_px = int(bar_len_px)
    if bar_px + 2 * m > Wc:
        if dashed_if_too_long:
            dashed = True
            bar_px = Wc - 2 * m
            if bar_px <= 0:
                return img_rgb01
        else:
            bar_px = Wc - 2 * m
            if bar_px <= 0:
                return img_rgb01

    # Draw
    if not dashed:
        x_end = Wc - m - 1
        x_start = max(0, x_end - bar_px + 1)
        img_rgb01[y_start : y_end + 1, x_start : x_end + 1, 0] = col[0]
        img_rgb01[y_start : y_end + 1, x_start : x_end + 1, 1] = col[1]
        img_rgb01[y_start : y_end + 1, x_start : x_end + 1, 2] = col[2]
        return img_rgb01

    # dashed
    start_x = m
    end_x = Wc - m - 1
    total = max(0, end_x - start_x + 1)
    if total <= 0:
        return img_rgb01
    segs = 2 * max(1, int(dash_count)) + 1
    group = total / segs
    dash_len = max(1, int(round(group)))
    for i in range(max(1, int(dash_count))):
        xs = int(round(start_x + group * (2 * i + 1)))
        xe = min(end_x, xs + dash_len - 1)
        if xs <= xe:
            img_rgb01[y_start : y_end + 1, xs : xe + 1, 0] = col[0]
            img_rgb01[y_start : y_end + 1, xs : xe + 1, 1] = col[1]
            img_rgb01[y_start : y_end + 1, xs : xe + 1, 2] = col[2]
    return img_rgb01


def get_latest_run_dir(classifier_output_dir, last_run_dir=None):
    """Get the most recent classifier run directory.

    Args:
        classifier_output_dir: Base classifier output directory
        last_run_dir: Optional path to a specific run directory

    Returns:
        Name of the run directory (e.g., 'run_20251216_121317')
    """
    if last_run_dir:
        return Path(last_run_dir).name

    classifier_base = Path(classifier_output_dir) / "classifier"
    if not classifier_base.exists():
        raise ValueError(
            "Classifier output directory does not exist. Please train a model first."
        )

    run_dirs = sorted(
        [
            d
            for d in classifier_base.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ],
        key=lambda x: x.name,
        reverse=True,
    )

    if not run_dirs:
        raise ValueError(
            "No training run found. Please train a model first or manually set MODEL_RUN_DIR."
        )

    return run_dirs[0].name
