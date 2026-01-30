"""Assemble plate-level HCS OME-NGFF zarr stores from per-tile zarr outputs.

Walks the module output directory (e.g., output/sbs/images/), discovers
per-tile zarr stores, and creates plate-level HCS zarr stores in an
hcs/ subdirectory. Each image type (aligned, peaks, nuclei, etc.) gets
its own plate-level store. Per-tile data is symlinked to avoid duplication.

HCS plate structure (OME-NGFF v0.5):
    output/{module}/hcs/{image_type}/{plate}.zarr/
        zarr.json               <- plate metadata (rows, columns, wells)
        {row}/
            {col}/
                zarr.json       <- well metadata (list of fields)
                {field}/  -> symlink to ../../images/{plate}/{well}/{tile}/{image_type}.zarr/

Usage:
    Called by finalize_hcs_* Snakemake rules after all tiles are written.
    Expects snakemake.params.images_dir pointing to the images/ directory.
"""

import json
import os
import re
from pathlib import Path


def split_well(well_str):
    """Split a well identifier like 'A1' into (row, col) -> ('A', '1').

    Handles multi-letter rows (e.g., 'AA1') and multi-digit columns (e.g., 'A12').
    """
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(well_str))
    if not match:
        raise ValueError(f"Cannot parse well identifier: '{well_str}'")
    return match.group(1), match.group(2)


def discover_zarr_stores(images_dir):
    """Walk the images directory and discover per-tile zarr stores.

    Expected structure: images_dir/{plate}/{well}/{tile}/{image_type}.zarr/

    Returns:
        dict: Nested dict of {image_type: {plate: {well: {tile: zarr_path}}}}
    """
    images_path = Path(images_dir)
    stores = {}

    if not images_path.exists():
        return stores

    for zarr_dir in sorted(images_path.rglob("*.zarr")):
        if not zarr_dir.is_dir():
            continue

        # Parse path: images_dir / plate / well / tile / image_type.zarr
        # Or for preprocess SBS: images_dir / plate / well / tile / cycle / image_type.zarr
        rel = zarr_dir.relative_to(images_path)
        parts = list(rel.parts)

        if len(parts) < 4:
            continue

        image_type = parts[-1].replace(".zarr", "")
        # Determine depth: plate/well/tile/type.zarr (4) or plate/well/tile/cycle/type.zarr (5)
        if len(parts) == 4:
            plate, well, tile = parts[0], parts[1], parts[2]
            group_key = image_type
        elif len(parts) == 5:
            plate, well, tile, cycle = parts[0], parts[1], parts[2], parts[3]
            group_key = f"{image_type}_cycle{cycle}"
        else:
            continue

        stores.setdefault(group_key, {}).setdefault(plate, {}).setdefault(well, {})[
            tile
        ] = str(zarr_dir)

    return stores


def write_zarr_v3_group_metadata(path):
    """Write a minimal zarr v3 group zarr.json file."""
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "zarr.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def write_plate_metadata(plate_zarr_path, wells_by_row_col):
    """Write HCS plate-level zarr.json with OME-NGFF plate metadata.

    Args:
        plate_zarr_path: Path to {plate}.zarr/ directory.
        wells_by_row_col: dict of {(row, col): well_str} for all wells in this plate.
    """
    plate_path = Path(plate_zarr_path)
    plate_path.mkdir(parents=True, exist_ok=True)

    rows = sorted(set(rc[0] for rc in wells_by_row_col.keys()))
    cols = sorted(set(rc[1] for rc in wells_by_row_col.keys()), key=lambda x: int(x))

    plate_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "plate": {
                "acquisitions": [{"id": 0, "name": "default"}],
                "columns": [{"name": c} for c in cols],
                "rows": [{"name": r} for r in rows],
                "wells": [
                    {
                        "path": f"{rc[0]}/{rc[1]}",
                        "rowIndex": rows.index(rc[0]),
                        "columnIndex": cols.index(rc[1]),
                    }
                    for rc in sorted(wells_by_row_col.keys())
                ],
            }
        },
    }

    with open(plate_path / "zarr.json", "w") as f:
        json.dump(plate_metadata, f, indent=2)


def write_well_metadata(well_path, field_indices):
    """Write HCS well-level zarr.json listing fields (tiles).

    Args:
        well_path: Path to {row}/{col}/ directory inside plate zarr.
        field_indices: sorted list of field index strings.
    """
    well_dir = Path(well_path)
    well_dir.mkdir(parents=True, exist_ok=True)

    well_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "well": {
                "images": [
                    {"path": str(idx), "acquisition": 0} for idx in field_indices
                ],
            }
        },
    }

    with open(well_dir / "zarr.json", "w") as f:
        json.dump(well_metadata, f, indent=2)


def assemble_hcs_plate(image_type, plate, wells_data, hcs_dir, images_dir):
    """Assemble one HCS plate zarr store from per-tile stores.

    Args:
        image_type: e.g., "aligned", "peaks", "nuclei"
        plate: plate identifier
        wells_data: dict of {well: {tile: zarr_path}}
        hcs_dir: root HCS output directory
        images_dir: root images directory (for computing relative symlink paths)
    """
    plate_zarr = Path(hcs_dir) / image_type / f"{plate}.zarr"

    # Collect well info
    wells_by_row_col = {}
    for well_str in wells_data:
        row, col = split_well(well_str)
        wells_by_row_col[(row, col)] = well_str

    # Write plate metadata
    write_plate_metadata(plate_zarr, wells_by_row_col)

    # Write row group metadata
    for row in sorted(set(rc[0] for rc in wells_by_row_col.keys())):
        write_zarr_v3_group_metadata(plate_zarr / row)

    # Write well metadata and field symlinks
    for (row, col), well_str in sorted(wells_by_row_col.items()):
        well_dir = plate_zarr / row / col
        tiles = wells_data[well_str]
        field_indices = sorted(tiles.keys())

        write_well_metadata(well_dir, field_indices)

        # Create symlinks from field index to per-tile zarr store
        for tile in field_indices:
            field_link = well_dir / str(tile)
            target = Path(tiles[tile])

            if field_link.exists() or field_link.is_symlink():
                os.remove(field_link)

            # Compute relative path from symlink location to target
            rel_target = os.path.relpath(target, field_link.parent)
            os.symlink(rel_target, field_link)


def main():
    """Main entry point for Snakemake script."""
    images_dir = snakemake.params.images_dir
    hcs_dir = snakemake.params.hcs_dir

    print(f"Discovering zarr stores in: {images_dir}")
    stores = discover_zarr_stores(images_dir)

    if not stores:
        print("No zarr stores found. Skipping HCS metadata generation.")
        # Touch the sentinel output
        Path(snakemake.output[0]).touch()
        return

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

    # Touch the sentinel output
    Path(snakemake.output[0]).touch()


main()
