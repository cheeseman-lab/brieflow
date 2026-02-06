"""HCS (High Content Screening) OME-NGFF plate assembly utilities.

Functions to discover per-tile zarr stores and assemble them into
plate-level HCS zarr stores with OME-NGFF v0.5 metadata and symlinks.
"""

import json
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# High-level API (used by Snakemake scripts)
# ---------------------------------------------------------------------------


def discover_zarr_stores(images_dir):
    """Walk the images directory and discover per-tile zarr stores.

    Expected structure: images_dir/{plate}/{well}/{tile}/{image_type}.zarr/
    Or for SBS: images_dir/{plate}/{well}/{tile}/{cycle}/{image_type}.zarr/

    Args:
        images_dir: Path to the images directory to scan.

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

        rel = zarr_dir.relative_to(images_path)
        parts = list(rel.parts)

        if len(parts) < 4:
            continue

        image_type = parts[-1].replace(".zarr", "")
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


def assemble_hcs_plate(image_type, plate, wells_data, hcs_dir, images_dir):
    """Assemble one HCS plate zarr store from per-tile stores.

    Creates plate-level zarr.json metadata, well-level metadata, and
    symlinks from field indices to per-tile zarr stores.

    Args:
        image_type: e.g., "aligned", "peaks", "nuclei"
        plate: Plate identifier.
        wells_data: dict of {well: {tile: zarr_path}}
        hcs_dir: Root HCS output directory.
        images_dir: Root images directory (for computing relative symlink paths).
    """
    plate_zarr = Path(hcs_dir) / image_type / f"{plate}.zarr"

    wells_by_row_col = {}
    for well_str in wells_data:
        row, col = _split_well(well_str)
        wells_by_row_col[(row, col)] = well_str

    _write_plate_metadata(plate_zarr, wells_by_row_col)

    for row in sorted(set(rc[0] for rc in wells_by_row_col.keys())):
        _write_zarr_v3_group_metadata(plate_zarr / row)

    for (row, col), well_str in sorted(wells_by_row_col.items()):
        well_dir = plate_zarr / row / col
        tiles = wells_data[well_str]
        field_indices = sorted(tiles.keys())

        _write_well_metadata(well_dir, field_indices)

        for tile in field_indices:
            field_link = well_dir / str(tile)
            target = Path(tiles[tile])

            if field_link.exists() or field_link.is_symlink():
                os.remove(field_link)

            rel_target = os.path.relpath(target, field_link.parent)
            os.symlink(rel_target, field_link)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_well(well_str):
    """Split a well identifier like 'A1' into (row, col) -> ('A', '1')."""
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(well_str))
    if not match:
        raise ValueError(f"Cannot parse well identifier: '{well_str}'")
    return match.group(1), match.group(2)


def _write_zarr_v3_group_metadata(path):
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


def _write_plate_metadata(plate_zarr_path, wells_by_row_col):
    """Write HCS plate-level zarr.json with OME-NGFF plate metadata."""
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


def _write_well_metadata(well_path, field_indices):
    """Write HCS well-level zarr.json listing fields (tiles)."""
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
