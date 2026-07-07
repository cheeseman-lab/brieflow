"""Lazy chunked fusion of tiles into an OME-Zarr v3 mosaic."""

from __future__ import annotations

import shutil

import numpy as np
import zarr


def fuse_mosaic(planes, offsets, out_path, chunk=1024, blend="linear"):
    """Fuse tile planes at global offsets into a chunked zarr v3 mosaic.

    Args:
        planes: {tile: 2D image}.
        offsets: TileOffsets in the global well frame.
        out_path: destination zarr path.
        chunk: chunk edge in px (bounds peak memory).
        blend: "linear" weighted overlap blending or "none".

    Returns:
        The written zarr path.
    """
    off = offsets.to_frame().set_index("tile")
    th, tw = next(iter(planes.values())).shape
    ys = {t: int(round(off.loc[t, "y"])) for t in planes}
    xs = {t: int(round(off.loc[t, "x"])) for t in planes}
    y0 = min(ys.values())
    x0 = min(xs.values())
    H = max(ys[t] - y0 + th for t in planes)
    W = max(xs[t] - x0 + tw for t in planes)

    acc_path = out_path + ".acc.tmp"
    wsum_path = out_path + ".w.tmp"
    acc = zarr.open(acc_path, mode="w", shape=(H, W),
                    chunks=(chunk, chunk), dtype="f4")
    wsum = zarr.open(wsum_path, mode="w", shape=(H, W),
                     chunks=(chunk, chunk), dtype="f4")
    weight = np.ones((th, tw), np.float32)
    if blend == "linear":
        wy = np.minimum(np.arange(th), np.arange(th)[::-1]) + 1.0
        wx = np.minimum(np.arange(tw), np.arange(tw)[::-1]) + 1.0
        weight = np.outer(wy, wx).astype(np.float32)
    for t, plane in planes.items():
        yy, xx = ys[t] - y0, xs[t] - x0
        acc[yy:yy + th, xx:xx + tw] = acc[yy:yy + th, xx:xx + tw] + plane * weight
        wsum[yy:yy + th, xx:xx + tw] = wsum[yy:yy + th, xx:xx + tw] + weight

    out = zarr.open(out_path, mode="w", shape=(H, W),
                    chunks=(chunk, chunk), dtype="f4")
    for i in range(0, H, chunk):
        for j in range(0, W, chunk):
            a = acc[i:i + chunk, j:j + chunk]
            w = wsum[i:i + chunk, j:j + chunk]
            out[i:i + chunk, j:j + chunk] = np.divide(a, w, out=np.zeros_like(a),
                                                       where=w > 0)

    shutil.rmtree(acc_path, ignore_errors=True)
    shutil.rmtree(wsum_path, ignore_errors=True)

    return out_path
