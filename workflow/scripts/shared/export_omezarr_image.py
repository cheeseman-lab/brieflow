import numpy as np
import pandas as pd
from tifffile import imread
from lib.shared.omezarr_writer import write_image_omezarr
from lib.preprocess.preprocess import convert_to_array

# TODO Add option to export zarr OR tiffs for downstream processing


def _read_image_data():
    """Read image data from TIFF or from original files via `convert_to_array`.

    `convert_to_array` is used when exporting with Z (preserve_z) and/or when inputs are
    non-TIFF (e.g. ND2).
    """
    data_format = snakemake.params.get("data_format", None)
    data_organization = snakemake.params.get("data_organization", None)
    preserve_z = bool(snakemake.params.get("preserve_z", False))
    channel_order_flip = bool(snakemake.params.get("channel_order_flip", False))

    def _is_tiff_path(p: str) -> bool:
        return str(p).lower().endswith((".tif", ".tiff"))

    def _is_nd2_path(p: str) -> bool:
        return str(p).lower().endswith(".nd2")

    # Normalize input(s)
    input_paths = snakemake.input.image
    if isinstance(input_paths, (list, tuple)):
        first_path = input_paths[0] if len(input_paths) else ""
    else:
        first_path = input_paths

    # Use convert_to_array when we are actually given raw files (e.g. ND2) or when we
    # explicitly want to preserve Z (which requires reading from the raw format).
    use_convert = (
        bool(preserve_z)
        or _is_nd2_path(first_path)
        or (
            isinstance(input_paths, (list, tuple))
            and any(_is_nd2_path(p) for p in input_paths)
        )
    )

    if use_convert and data_format is not None and data_organization is not None:
        position = None
        if data_organization == "well":
            position = int(snakemake.params.get("tile", 0))

        return convert_to_array(
            snakemake.input.image,
            data_format=data_format,
            data_organization=data_organization,
            position=position,
            channel_order_flip=channel_order_flip,
            preserve_z=preserve_z,
            verbose=False,
        )

    # Default: read the pre-converted TIFF
    return imread(snakemake.input.image)


# Read input image
image_data = _read_image_data()

# Ensure correct dimensions
# tifffile might return (Y, X) for 2D, (C, Y, X) for 3D, or (C, Z, Y, X) for 4D.
if image_data.ndim == 2:
    image_data = image_data[np.newaxis, :, :]

# Extract pixel size from metadata if available
pixel_size_um = None
if hasattr(snakemake.input, "metadata") and hasattr(snakemake.params, "tile"):
    try:
        # Read parquet metadata
        meta_df = pd.read_parquet(snakemake.input.metadata)

        # Filter for current tile
        tile_id = int(snakemake.params.tile)
        tile_meta = meta_df[meta_df["tile"] == tile_id]

        if not tile_meta.empty:
            px = (
                tile_meta["pixel_size_x"].iloc[0]
                if "pixel_size_x" in tile_meta.columns
                else None
            )
            py = (
                tile_meta["pixel_size_y"].iloc[0]
                if "pixel_size_y" in tile_meta.columns
                else None
            )
            pz = (
                tile_meta["pixel_size_z"].iloc[0]
                if "pixel_size_z" in tile_meta.columns
                else None
            )

            # Apply upsample factor (spatial X/Y only)
            upsample = float(snakemake.params.get("upsample_factor", 1.0))
            if upsample <= 0:
                upsample = 1.0

            def _valid(v):
                return v is not None and pd.notna(v) and float(v) > 0

            px_u = float(px) / upsample if _valid(px) else None
            py_u = (
                float(py) / upsample
                if _valid(py)
                else (px_u if px_u is not None else None)
            )
            pz_u = float(pz) if _valid(pz) else None

            # Provide per-axis calibration when available; otherwise fall back to scalar.
            if px_u is not None or py_u is not None or pz_u is not None:
                pixel_size_um = {"x": px_u, "y": py_u, "z": pz_u}
    except Exception as e:
        print(f"Warning: Failed to extract pixel size from metadata: {e}")

# Write OME-Zarr
write_image_omezarr(
    image_data=image_data,
    out_path=snakemake.output[0],
    axes=snakemake.params.axes,
    channel_names=snakemake.params.get("channel_names"),
    pixel_size_um=pixel_size_um,
)
