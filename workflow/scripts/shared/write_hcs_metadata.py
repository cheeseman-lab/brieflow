"""Write HCS plate-level OME-NGFF metadata for plate zarr directories.

For each plate zarr dir: write plate/row/well/labels metadata, then the
per-field metadata (pixel sizes, axis units, OMERO rendering, channel names,
label annotations).

After all plates are written, compute the screen-wide per-channel intensity
percentiles (1st/99th) and summary statistics (mean/std/median) and write them
into every field zarr.json under ``omero.channels[i]``.
"""

from pathlib import Path

from lib.shared.hcs import (
    write_channel_intensity_statistics,
    write_field_image_metadata,
    write_hcs_metadata,
)


plate_zarr_dirs = snakemake.params.plate_zarr_dirs
channels_metadata = getattr(snakemake.params, "channels_metadata", None)
config_channel_names = getattr(snakemake.params, "channel_names", None)

# Worker threads for the per-field pixel reads (histograms, object counts)
threads = getattr(snakemake, "threads", 1) or 1

# Preprocess root for pixel-size lookup
root_fp = Path(snakemake.config["all"]["root_fp"])
preprocess_root = root_fp / "preprocess"

# Modality config (params.modality: sbs | phenotype) for label segmentation_metadata
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
        # Skip preprocess stores (extra cycle nesting)
        if "preprocess" not in plate_path.parts:
            write_field_image_metadata(
                plate_path,
                preprocess_root,
                config_channel_names=config_channel_names,
                modality_config=modality_config,
                channels_metadata=channels_metadata,
                threads=threads,
            )
    else:
        print(f"Plate zarr not found, skipping: {plate_path}")

if total > 0:
    print(f"\nHCS metadata written for {total} plate zarr(s).")
else:
    print("No plate zarr directories found. Skipping HCS metadata.")

# Screen-wide intensity percentiles and statistics; preprocess stores need no rendering
renderable_plates = [
    Path(p)
    for p in plate_zarr_dirs
    if Path(p).exists() and "preprocess" not in Path(p).parts
]
write_channel_intensity_statistics(renderable_plates, threads=threads)
