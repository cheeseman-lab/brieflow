"""Pseudotile merge: per-SBS-tile phenotype recut + hash (correspondence by physical position)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def register_stage_frames(
    sbs_meta: pd.DataFrame,
    ph_meta: pd.DataFrame,
    x_col: str = "x_pos",
    y_col: str = "y_pos",
) -> dict:
    """Estimate the translation aligning the phenotype stage frame onto the SBS stage frame.

    Both metadata tables carry per-tile stage coordinates (µm) in a shared physical
    coordinate system that differs only by a fixed origin offset. The aligning transform
    is the vector that maps a phenotype stage coordinate onto the SBS stage frame: the
    difference of the two frames' robust (median) centers.

    Args:
        sbs_meta: SBS per-tile metadata with ``x_col``/``y_col`` stage coords (µm).
        ph_meta: Phenotype per-tile metadata with the same columns.
        x_col: stage-x column name.
        y_col: stage-y column name.

    Returns:
        Dict with:
            translation (np.ndarray, shape (2,)): (dy, dx) in µm added to a phenotype
                stage coordinate (y, x) to bring it into the SBS stage frame.
            rotation (np.ndarray, shape (2,2)): identity (frames differ by translation only).
    """
    translation = np.array(
        [
            float(np.median(sbs_meta[y_col])) - float(np.median(ph_meta[y_col])),
            float(np.median(sbs_meta[x_col])) - float(np.median(ph_meta[x_col])),
        ]
    )
    rotation = np.eye(2)
    return {"translation": translation, "rotation": rotation}
