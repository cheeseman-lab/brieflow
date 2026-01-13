"""Simple script to load BrieFlow OME-Zarr in Napari (for Jupyter notebooks).

Copy and paste this into your notebook.
"""

import napari
import zarr
import numpy as np
from pathlib import Path

# Change this to your zarr path
zarr_path = "/path/to/your/file.zarr"  # UPDATE THIS PATH

# Validate path
zarr_path = Path(zarr_path)
if not zarr_path.exists():
    raise FileNotFoundError(
        f"OME-Zarr path does not exist: {zarr_path}\n"
        f"Please update the zarr_path variable with the correct path."
    )

if not zarr_path.is_dir():
    raise ValueError(
        f"Path is not a directory: {zarr_path}\n"
        f"OME-Zarr files should be directories with .zarr extension."
    )

# Create viewer
viewer = napari.Viewer()

# Open zarr
try:
    root = zarr.open_group(str(zarr_path), mode="r")
except Exception as e:
    raise ValueError(
        f"Failed to open zarr group: {zarr_path}\n"
        f"Error: {e}\n"
        f"This may not be a valid Zarr file."
    )

# Check for required scale
if "0" not in root:
    raise ValueError(
        f"Scale '0' not found in {zarr_path}\n"
        f"OME-Zarr files must contain at least one resolution level named '0'."
    )

# Load image (highest resolution = scale 0)
try:
    image = root["0"][:]
    print(f"Image shape: {image.shape}")
except Exception as e:
    raise ValueError(f"Failed to load image data from {zarr_path}\nError: {e}")

# Get pixel size from metadata
pixel_size = root.attrs.get("omero", {}).get("pixel_size", {})
px = pixel_size.get("x", 1.0)  # default to 1.0 if not specified
py = pixel_size.get("y", 1.0)

# Get axes info to determine channel axis
multiscales = root.attrs.get("multiscales", [{}])[0]
axes_info = multiscales.get("axes", [])
axes_names = [ax["name"] for ax in axes_info]
channel_axis = next((i for i, ax in enumerate(axes_names) if ax == "c"), None)

# Determine if multichannel
if channel_axis is not None:
    n_channels = image.shape[channel_axis]
    print(f"Loading as multichannel image with {n_channels} channels")
    print(f"Axes: {axes_names}")

    # Generate colormaps for all channels
    base_colormaps = ["gray", "green", "red", "cyan", "magenta", "yellow", "blue"]
    colormaps = [base_colormaps[i % len(base_colormaps)] for i in range(n_channels)]

    viewer.add_image(
        image,
        name=zarr_path.stem,
        channel_axis=channel_axis,
        scale=[py, px],  # Y, X only (channel_axis handles C dimension)
        colormap=colormaps,
        blending="additive",
        contrast_limits=[
            (0, image.take(i, axis=channel_axis).max()) for i in range(n_channels)
        ],
    )
else:
    # Single channel
    print(f"Loading as single channel image")
    viewer.add_image(image, name=zarr_path.stem, scale=[py, px], colormap="gray")

# Load labels (if they exist)
labels_path = zarr_path / "labels"
if labels_path.exists():
    try:
        labels_group = zarr.open_group(str(labels_path), mode="r")
        label_names = labels_group.attrs.get("labels", [])

        print(f"\nFound {len(label_names)} label groups: {label_names}")

        for label_name in label_names:
            try:
                label_data = labels_group[label_name]["0"][:]
                n_objects = len(np.unique(label_data)) - 1  # subtract background
                print(f"  Loading '{label_name}': {n_objects} objects")

                viewer.add_labels(
                    label_data,
                    name=label_name,
                    scale=[py, px],  # Y, X only for labels
                )
            except Exception as e:
                print(f"  Warning: Could not load label '{label_name}': {e}")
    except Exception as e:
        print(f"Warning: Could not load labels: {e}")
else:
    print("\nNo labels found in this OME-Zarr.")

print("\nSuccessfully loaded! Close the Napari window when finished.")
