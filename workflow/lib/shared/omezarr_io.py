"""OME-Zarr IO helpers kept for backwards-compatible imports in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .omezarr_writer import write_image_omezarr


def write_multiscale_omezarr(
    *,
    image: np.ndarray,
    output_dir: Union[str, Path],
    coarsening_factor: int = 2,
    max_levels: int = 5,
    pixel_size: Optional[Union[float, Tuple[float, ...]]] = None,
    channel_names: Optional[Sequence[str]] = None,
    is_label: bool = False,
) -> None:
    """Write an OME-NGFF v0.4 (Zarr v2) multiscale pyramid.

    This is a small wrapper around `workflow.lib.shared.omezarr_writer.write_image_omezarr`.
    """
    out = Path(output_dir)

    if image.ndim == 2:
        data = image[np.newaxis, ...]
        axes = "cyx"
    elif image.ndim == 3:
        data = image
        axes = "cyx"
    elif image.ndim == 4:
        data = image
        axes = "czyx"
    else:
        raise ValueError(f"Unsupported image.ndim={image.ndim} for OME-Zarr export")

    write_image_omezarr(
        image_data=data,
        out_path=str(out),
        channel_names=(
            list(channel_names)
            if channel_names is not None
            else [f"c{i}" for i in range(int(data.shape[axes.index("c")]))]
        )
        if "c" in axes
        else None,
        axes=axes,
        pixel_size_um=pixel_size,
        coarsening_factor=coarsening_factor,
        max_levels=max_levels,
        is_label=is_label,
    )
