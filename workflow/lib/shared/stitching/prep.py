"""Tile-image preprocessing for registration."""

from __future__ import annotations

import numpy as np


def select_registration_plane(
    image: np.ndarray, channel: int, cycle: int | None
) -> np.ndarray:
    """Reduce a tile stack to the 2D plane used for registration.

    Args:
        image: Tile array. 4D is (cycle, channel, y, x); 3D is (channel, y, x).
        channel: Channel index to register on (e.g. DAPI).
        cycle: Cycle index for 4D stacks; None for 3D stacks.

    Returns:
        A 2D float32 image plane.
    """
    arr = image
    if arr.ndim == 4:
        if cycle is None:
            raise ValueError("cycle is required for a 4D tile stack")
        arr = arr[cycle, channel]
    elif arr.ndim == 3:
        arr = arr[channel]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected image dimensions: {arr.shape}")
    return np.asarray(arr, dtype=np.float32)
