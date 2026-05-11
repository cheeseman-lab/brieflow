"""Write HCS plate-level OME-NGFF metadata for plate zarr directories.

For each plate zarr dir: write plate/row/well/labels metadata, enrich
tile-level metadata via iohub (pixel sizes, axis units, OMERO rendering,
channel names), and re-inject downsamplingMethod that iohub strips.

After all plates are written, compute screen-wide per-channel display
windows (1st/99th percentile) and per-channel statistics (mean/std/median),
then inject them into every tile zarr.json under ``omero.channels[i]``.
"""

from pathlib import Path

from lib.shared.hcs import (
    compute_and_inject_omero_windows,
    patch_store_metadata_with_iohub,
    write_hcs_metadata,
)


plate_zarr_dirs = snakemake.params.plate_zarr_dirs
channels_metadata = getattr(snakemake.params, "channels_metadata", None)
config_channel_names = getattr(snakemake.params, "channel_names", None)

# Preprocess root for pixel-size lookup
root_fp = Path(snakemake.config["all"]["root_fp"])
preprocess_root = root_fp / "preprocess"

# Modality config for segmentation_metadata on label stores.
# The snakemake rule should set params.modality to "sbs" or "phenotype".
modality_name = getattr(snakemake.params, "modality", None)
modality_config = None
if modality_name and modality_name in snakemake.config:
    modality_config = dict(snakemake.config[modality_name])

total = 0
for plate_zarr in plate_zarr_dirs:
    plate_path = Path(plate_zarr)
    if plate_path.exists():
        print(f"Writing HCS metadata for: {plate_path}")
        write_hcs_metadata(plate_path, channels_metadata=channels_metadata)
        total += 1
        # Skip preprocess stores for iohub patching (extra cycle nesting)
        if "preprocess" not in plate_path.parts:
            patch_store_metadata_with_iohub(
                plate_path,
                preprocess_root,
                config_channel_names=config_channel_names,
                modality_config=modality_config,
                channels_metadata=channels_metadata,
            )
    else:
        print(f"Plate zarr not found, skipping: {plate_path}")

if total > 0:
    print(f"\nHCS metadata written for {total} plate zarr(s).")
else:
    print("No plate zarr directories found. Skipping HCS metadata.")

# Screen-wide OMERO display window + per-channel statistics.
# Skip preprocess stores (different cycle nesting, no rendering need).
renderable_plates = [
    Path(p)
    for p in plate_zarr_dirs
    if Path(p).exists() and "preprocess" not in Path(p).parts
]
compute_and_inject_omero_windows(renderable_plates)
