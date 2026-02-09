#!/usr/bin/env python3
"""Load BrieFlow OME-Zarr outputs in Napari.

Handles multichannel images and labels. Reads pixel sizes, channel names,
and contrast limits from OME-NGFF v0.5 metadata when available.

Usage:
    python load_omezarr_in_napari.py <path_to.zarr>

    # Per-tile store (with labels if present):
    python load_omezarr_in_napari.py output/sbs/images/1/A1/0/aligned.zarr

    # HCS field via plate zarr (labels nested under labels/):
    python load_omezarr_in_napari.py output/sbs/hcs/1.zarr/A/1/0

Requirements:
    conda create -n napari-viz -c conda-forge python=3.11 napari zarr numpy -y
"""

import sys
from pathlib import Path

import napari
import numpy as np
import zarr


def load_omezarr_to_napari(zarr_path: str) -> napari.Viewer:
    """Load an OME-Zarr store (image + labels) into a Napari viewer.

    Args:
        zarr_path: Path to a .zarr directory or HCS field directory.

    Returns:
        The Napari Viewer instance with layers added.
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Path does not exist: {zarr_path}")
    if not zarr_path.is_dir():
        raise ValueError(f"Not a directory: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="r")

    # Read OME multiscale metadata
    multiscales = root.attrs.get("multiscales", [{}])[0]
    axes_info = multiscales.get("axes", [])
    axes_names = [ax["name"] for ax in axes_info]
    if not axes_names:
        raise ValueError(f"No axes metadata in {zarr_path}")

    image_data = root["0"][:]

    # Pixel sizes from omero metadata (populated after enrichment)
    pixel_size = root.attrs.get("omero", {}).get("pixel_size", {})
    channel_axis = next((i for i, ax in enumerate(axes_names) if ax == "c"), None)

    # Build spatial scale (excluding channel axis)
    scale = []
    for i, ax in enumerate(axes_names):
        if i == channel_axis:
            continue
        if ax in ("x", "y") and pixel_size:
            scale.append(pixel_size.get(ax, 1.0))
        else:
            scale.append(1.0)

    viewer = napari.Viewer()

    if channel_axis is not None:
        n_channels = image_data.shape[channel_axis]

        # Channel names from omero metadata
        omero_channels = root.attrs.get("omero", {}).get("channels", [])
        channel_names = (
            [ch.get("label", f"Channel {i}") for i, ch in enumerate(omero_channels)]
            if omero_channels
            else [f"Channel {i}" for i in range(n_channels)]
        )

        # Contrast limits: use omero window if available, else percentile
        contrast_limits = []
        for i in range(n_channels):
            if i < len(omero_channels) and "window" in omero_channels[i]:
                w = omero_channels[i]["window"]
                contrast_limits.append((w["start"], w["end"]))
            else:
                ch_data = np.take(image_data, i, axis=channel_axis)
                contrast_limits.append(
                    (
                        np.percentile(ch_data, 1),
                        np.percentile(ch_data, 99.5),
                    )
                )

        base_colormaps = ["gray", "green", "red", "cyan", "magenta", "yellow", "blue"]
        colormaps = [base_colormaps[i % len(base_colormaps)] for i in range(n_channels)]

        print(f"Loading {n_channels} channels: {channel_names}")
        print(f"  shape={image_data.shape}, scale={scale}")

        viewer.add_image(
            image_data,
            name=zarr_path.stem,
            channel_axis=channel_axis,
            scale=scale,
            colormap=colormaps,
            blending="additive",
            contrast_limits=contrast_limits,
            interpolation2d="nearest",
        )
    else:
        vmin = np.percentile(image_data, 1)
        vmax = np.percentile(image_data, 99.5)
        print(f"Loading single-channel image: shape={image_data.shape}, scale={scale}")

        viewer.add_image(
            image_data,
            name=zarr_path.stem,
            scale=scale,
            colormap="gray",
            contrast_limits=(vmin, vmax),
            interpolation2d="nearest",
        )

    # Load labels if present (standalone store or HCS field)
    labels_path = zarr_path / "labels"
    if labels_path.exists():
        labels_group = zarr.open_group(str(labels_path), mode="r")
        label_names = labels_group.attrs.get("labels", [])
        print(f"Loading {len(label_names)} label layers: {label_names}")

        for label_name in label_names:
            try:
                label_group = labels_group[label_name]
                label_data = label_group["0"][:]

                # Match label scale to image scale
                label_ms = label_group.attrs.get("multiscales", [{}])[0]
                label_axes = [ax["name"] for ax in label_ms.get("axes", [])]
                label_scale = [
                    pixel_size.get(ax, 1.0) if ax in ("x", "y") and pixel_size else 1.0
                    for ax in label_axes
                ]

                n_objects = len(np.unique(label_data)) - 1
                print(f"  {label_name}: {n_objects} objects, shape={label_data.shape}")
                viewer.add_labels(label_data, name=label_name, scale=label_scale)
            except Exception as e:
                print(f"  Warning: could not load '{label_name}': {e}")

    print(f"\nLoaded {zarr_path.name}")
    return viewer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_omezarr_in_napari.py <path_to.zarr>")
        sys.exit(1)

    viewer = load_omezarr_to_napari(sys.argv[1])
    napari.run()
