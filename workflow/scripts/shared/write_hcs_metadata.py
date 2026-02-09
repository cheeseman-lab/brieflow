"""Write HCS plate-level OME-NGFF metadata for plate zarr directories.

This script discovers the structure of plate zarr directories that were
populated directly by Snakemake jobs and writes the necessary zarr.json
metadata files at plate, row, well, and labels levels.
"""

from pathlib import Path

from lib.shared.hcs import write_hcs_metadata

plate_zarr_dirs = snakemake.params.plate_zarr_dirs

total = 0
for plate_zarr in plate_zarr_dirs:
    plate_path = Path(plate_zarr)
    if plate_path.exists():
        print(f"Writing HCS metadata for: {plate_path}")
        write_hcs_metadata(plate_path)
        total += 1
    else:
        print(f"Plate zarr not found, skipping: {plate_path}")

if total > 0:
    print(f"\nHCS metadata written for {total} plate zarr(s).")
else:
    print("No plate zarr directories found. Skipping HCS metadata.")
