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


def stage_coarse_transform(
    sbs_meta: pd.DataFrame,
    ph_meta: pd.DataFrame,
    sbs_um_per_px: float,
    ph_um_per_px: float,
    x_col: str = "x_pos",
    y_col: str = "y_pos",
) -> dict:
    """Build a translation-only coarse transform mapping PH global px -> SBS global px.

    Uses ``register_stage_frames`` to align the two modalities' stage frames by a pure
    translation, then composes with each modality's stitch-prior origin
    (``meta[y_col].min()`` / ``meta[x_col].min()`` — the ``y_min``/``x_min`` used when the
    stitch prior was built) and the pixel-size ratio, yielding an affine of the form
    ``ph_ref = (scale · I) @ ph_global_px + t_px`` with NO rotation. The returned dict has
    the same keys as ``coarse_align_dapi`` so it drops straight into
    ``merge_reference_tiles_per_phtile``.

    Derivation: a PH cell at global px ``g_ph`` has absolute PH-stage µm
    ``g_ph·ph_um_per_px + ph_min``. Adding the inter-frame translation ``t`` brings it into
    the SBS stage frame; subtracting ``sbs_min`` and dividing by ``sbs_um_per_px`` converts
    to SBS global px: ``scale·g_ph + (ph_min + t − sbs_min)/sbs_um_per_px``.

    Args:
        sbs_meta: SBS per-tile metadata with stage coords (µm).
        ph_meta: Phenotype per-tile metadata with stage coords (µm).
        sbs_um_per_px: SBS pixel size (µm/px).
        ph_um_per_px: Phenotype pixel size (µm/px).
        x_col: stage-x column.
        y_col: stage-y column.

    Returns:
        Dict with keys ``rotation`` (2x2 = scale·I), ``translation`` ((2,) SBS px),
        ``angle_deg`` (0.0), ``scale`` (ph_um_per_px / sbs_um_per_px).
    """
    stage = register_stage_frames(sbs_meta, ph_meta, x_col=x_col, y_col=y_col)
    ty, tx = float(stage["translation"][0]), float(stage["translation"][1])
    scale = ph_um_per_px / sbs_um_per_px
    ph_y_min = float(ph_meta[y_col].min())
    ph_x_min = float(ph_meta[x_col].min())
    sbs_y_min = float(sbs_meta[y_col].min())
    sbs_x_min = float(sbs_meta[x_col].min())
    const_y = (ph_y_min + ty - sbs_y_min) / sbs_um_per_px
    const_x = (ph_x_min + tx - sbs_x_min) / sbs_um_per_px
    return {
        "rotation": scale * np.eye(2),
        "translation": np.array([const_y, const_x], dtype=np.float64),
        "angle_deg": 0.0,
        "scale": scale,
    }
