"""Utilities for working with OME-Zarr stores (metadata, colors, labels).

This module is intentionally lightweight and does **not** wire itself into the
Snakemake rules. You can:

- Call these helpers manually to patch existing `.ome.zarr` stores.
- Or import them from writer code (e.g. an ND2→OME-Zarr converter) to ensure
  that the written metadata is napari/OME-NGFF friendly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, Any

import json

import numpy as np
import zarr


PathLike = Union[str, Path]


def _hex_to_omero_int(hex_color: str, alpha: int = 255) -> int:
    """Convert a '#RRGGBB' hex string to an OMERO ARGB integer."""
    hex_color = hex_color.strip()
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color!r}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(alpha) & 0xFF

    # OMERO uses ARGB stored in a single 32-bit integer
    return (a << 24) | (r << 16) | (g << 8) | b


def _rgba_tuple_to_omero_int(
    rgba: Sequence[Union[int, float]], assume_0_1: bool = False
) -> int:
    """Convert (R, G, B) or (R, G, B, A) tuple to OMERO ARGB integer."""
    if len(rgba) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 values for RGBA, got {rgba!r}")

    if assume_0_1:
        vals = [int(round(float(c) * 255)) for c in rgba]
    else:
        vals = [int(round(float(c))) for c in rgba]

    if len(vals) == 3:
        r, g, b = vals
        a = 255
    else:
        r, g, b, a = vals

    r &= 0xFF
    g &= 0xFF
    b &= 0xFF
    a &= 0xFF

    return (a << 24) | (r << 16) | (g << 8) | b


def _normalize_color_to_omero_int(color: Any) -> Optional[int]:
    """Normalize various color representations to an OMERO ARGB integer.

    Accepted inputs:
    - int: assumed to already be an OMERO ARGB integer, returned as-is.
    - '#RRGGBB' hex string.
    - (R, G, B) or (R, G, B, A) tuples/lists with values in [0, 255] or [0, 1].

    Returns:
        OMERO ARGB integer, or None if color is None/unknown.
    """
    if color is None:
        return None

    # Already an integer
    if isinstance(color, int):
        return int(color)

    # Hex string
    if isinstance(color, str):
        return _hex_to_omero_int(color)

    # Tuple/list
    if isinstance(color, (list, tuple, np.ndarray)):
        arr = np.asarray(color, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Unsupported color array shape: {arr.shape}")

        # Heuristic: values in [0, 1] vs [0, 255]
        assume_0_1 = bool(np.all((arr >= 0.0) & (arr <= 1.0)))
        return _rgba_tuple_to_omero_int(arr, assume_0_1=assume_0_1)

    raise ValueError(f"Unsupported color type: {type(color)!r}")


def _default_hex_palette(n: int) -> List[str]:
    """Return a repeating default hex palette of length n."""
    base_palette = [
        "#0000FF",  # blue
        "#00FF00",  # green
        "#FF0000",  # red
        "#FFFF00",  # yellow
        "#FF00FF",  # magenta
        "#00FFFF",  # cyan
        "#FFFFFF",  # white
    ]
    if n <= len(base_palette):
        return base_palette[:n]
    reps = (n + len(base_palette) - 1) // len(base_palette)
    return (base_palette * reps)[:n]


def ensure_omero_channel_colors(
    zarr_path: PathLike,
    channel_labels: Optional[Sequence[str]] = None,
    channel_colors: Optional[Sequence[Any]] = None,
) -> None:
    """Ensure `omero.channels` exists and has colors for an OME-Zarr store.

    This is designed to be run:
    - After writing a new OME-Zarr store, or
    - On existing stores that are missing channel color metadata.

    It will:
    - Inspect `scale0` to determine the number of channels.
    - Create or update `attrs['omero']['channels']` so that:
        - Its length matches the number of channels.
        - Each entry has a `label` and `color` field.
    - Colors are taken from `channel_colors` if provided, otherwise from a
      default palette (converted to OMERO ARGB integers).

    Args:
        zarr_path: Path to the `.ome.zarr` directory.
        channel_labels: Optional labels per channel (len == n_channels).
        channel_colors: Optional colors per channel; entries can be:
            - OMERO ARGB integers,
            - '#RRGGBB' hex strings, or
            - (R, G, B[, A]) tuples/lists in [0, 255] or [0, 1].
    """
    zarr_path = Path(zarr_path)

    store = zarr_path
    group = zarr.open(store=str(store), mode="r")

    if "scale0" not in group:
        raise ValueError(f"No 'scale0' dataset found in {zarr_path}")

    scale0 = group["scale0"]
    if scale0.ndim < 3:
        raise ValueError(
            f"Expected at least 3D data in scale0 (C, Y, X), got shape {scale0.shape}"
        )

    n_channels = int(scale0.shape[0])

    attrs_path = zarr_path / ".zattrs"
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing .zattrs in {zarr_path}")

    attrs = json.loads(attrs_path.read_text())

    # Ensure top-level omero key
    omero = attrs.get("omero") or {}

    # Base channel list if present
    channels_meta: List[dict] = list(omero.get("channels") or [])

    # Normalize lengths
    if len(channels_meta) < n_channels:
        # Extend with empty dicts
        channels_meta.extend({} for _ in range(n_channels - len(channels_meta)))
    elif len(channels_meta) > n_channels:
        # Truncate extra entries
        channels_meta = channels_meta[:n_channels]

    # Prepare defaults
    default_hex = _default_hex_palette(n_channels)

    for idx in range(n_channels):
        ch = channels_meta[idx] or {}

        # Label
        if channel_labels is not None and idx < len(channel_labels):
            label = channel_labels[idx]
        else:
            # Preserve existing label if present, fallback otherwise
            label = ch.get("label") or f"Channel-{idx}"
        ch["label"] = label

        # Color
        raw_color = None
        if channel_colors is not None and idx < len(channel_colors):
            raw_color = channel_colors[idx]
        elif "color" in ch:
            raw_color = ch["color"]
        else:
            raw_color = default_hex[idx]

        try:
            color_int = _normalize_color_to_omero_int(raw_color)
        except ValueError:
            # Fallback to palette if normalization fails
            color_int = _normalize_color_to_omero_int(default_hex[idx])

        if color_int is not None:
            ch["color"] = int(color_int)

        channels_meta[idx] = ch

    omero["channels"] = channels_meta
    attrs["omero"] = omero

    attrs_path.write_text(json.dumps(attrs, indent=2))


