"""Assemble plate-level HCS OME-NGFF zarr stores from per-tile zarr outputs."""

from pathlib import Path

from lib.shared.hcs import assemble_hcs_plate, discover_zarr_stores

images_dir = snakemake.params.images_dir
hcs_dir = snakemake.params.hcs_dir

print(f"Discovering zarr stores in: {images_dir}")
stores = discover_zarr_stores(images_dir)

if not stores:
    print("No zarr stores found. Skipping HCS metadata generation.")
    Path(snakemake.output[0]).touch()
else:
    total_plates = 0
    for image_type, plates in sorted(stores.items()):
        for plate, wells_data in sorted(plates.items()):
            n_wells = len(wells_data)
            n_tiles = sum(len(tiles) for tiles in wells_data.values())
            print(
                f"  Assembling HCS: {image_type}/{plate}.zarr ({n_wells} wells, {n_tiles} tiles)"
            )
            assemble_hcs_plate(image_type, plate, wells_data, hcs_dir, images_dir)
            total_plates += 1

    print(f"HCS assembly complete: {total_plates} plate stores created in {hcs_dir}")
    Path(snakemake.output[0]).touch()
