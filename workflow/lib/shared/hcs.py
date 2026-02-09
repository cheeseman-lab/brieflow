"""HCS (High Content Screening) OME-NGFF plate assembly utilities.

Functions to discover per-tile zarr stores and assemble them into
plate-level HCS zarr stores with OME-NGFF v0.5 metadata and symlinks.
"""

import json
import os
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# High-level API (used by Snakemake scripts)
# ---------------------------------------------------------------------------


def discover_zarr_stores(images_dir):
    """Walk the images directory and discover per-tile zarr stores.

    Expected structure: images_dir/{plate}/{well}/{tile}/{image_type}.zarr/
    Or for SBS: images_dir/{plate}/{well}/{tile}/{cycle}/{image_type}.zarr/

    Stores are classified as images or labels by checking for the
    ``"image-label"`` key in each store's ``zarr.json`` attributes.

    Args:
        images_dir: Path to the images directory to scan.

    Returns:
        tuple: (images, labels) where each is
               {image_type: {plate: {well: {tile: zarr_path}}}}
    """
    images_path = Path(images_dir)
    images, labels = {}, {}

    if not images_path.exists():
        return images, labels

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

        target = labels if _is_label_store(zarr_dir) else images
        target.setdefault(group_key, {}).setdefault(plate, {}).setdefault(well, {})[
            tile
        ] = str(zarr_dir)

    return images, labels


def assemble_hcs_plate(plate, images_by_type, hcs_dir, images_dir, labels_data=None):
    """Assemble one HCS plate zarr store containing all image types.

    The first image type (alphabetically) becomes the primary multiscale
    image at each field.  Other intensity image types are nested as named
    subgroups inside the field directory.  Label stores are nested under a
    ``labels/`` subgroup per the OME-NGFF spec.

    Args:
        plate: Plate identifier.
        images_by_type: dict of {image_type: {well: {tile: zarr_path}}}
        hcs_dir: Root HCS output directory.
        images_dir: Root images directory (for computing relative symlink paths).
        labels_data: Optional dict of {label_type: {plate: {well: {tile: zarr_path}}}}
    """
    plate_zarr = Path(hcs_dir) / f"{plate}.zarr"

    # Choose primary image type (first alphabetically)
    primary_type = sorted(images_by_type.keys())[0]
    primary_wells = images_by_type[primary_type]

    wells_by_row_col = {}
    for well_str in primary_wells:
        row, col = _split_well(well_str)
        wells_by_row_col[(row, col)] = well_str

    _write_plate_metadata(plate_zarr, wells_by_row_col)

    for row in sorted(set(rc[0] for rc in wells_by_row_col.keys())):
        _write_zarr_v3_group_metadata(plate_zarr / row)

    for (row, col), well_str in sorted(wells_by_row_col.items()):
        well_dir = plate_zarr / row / col
        primary_tiles = primary_wells[well_str]
        field_indices = sorted(primary_tiles.keys())

        _write_well_metadata(well_dir, field_indices)

        for tile in field_indices:
            field_dir = well_dir / str(tile)
            source_store = Path(primary_tiles[tile])

            # Create field as a real directory (not a whole-store symlink)
            if field_dir.is_symlink():
                os.remove(field_dir)
            field_dir.mkdir(parents=True, exist_ok=True)

            # Primary image: copy zarr.json and symlink pyramid levels
            shutil.copy2(source_store / "zarr.json", field_dir / "zarr.json")
            _symlink_pyramid_levels(source_store, field_dir)

            # Nest labels
            if labels_data:
                _nest_labels(field_dir, tile, labels_data, plate, well_str)

            # Nest other image types as named subgroups
            for img_type in sorted(images_by_type.keys()):
                if img_type == primary_type:
                    continue
                wells = images_by_type[img_type]
                if well_str not in wells or tile not in wells[well_str]:
                    continue
                other_store = Path(wells[well_str][tile])
                _nest_image_subgroup(field_dir, img_type, other_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_well(well_str):
    """Split a well identifier like 'A1' into (row, col) -> ('A', '1')."""
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(well_str))
    if not match:
        raise ValueError(f"Cannot parse well identifier: '{well_str}'")
    return match.group(1), match.group(2)


def _is_label_store(zarr_path):
    """Check if a zarr store is a label image by reading its zarr.json."""
    zarr_json = Path(zarr_path) / "zarr.json"
    if not zarr_json.exists():
        return False
    with open(zarr_json) as f:
        meta = json.load(f)
    return "image-label" in meta.get("attributes", {})


def _symlink_pyramid_levels(source_store, target_dir):
    """Create symlinks for each pyramid level directory in source_store."""
    for level_dir in sorted(source_store.iterdir()):
        if level_dir.is_dir() and level_dir.name.isdigit():
            link = target_dir / level_dir.name
            if link.exists() or link.is_symlink():
                os.remove(link)
            os.symlink(os.path.relpath(level_dir, target_dir), link)


def _nest_labels(field_dir, tile, labels_data, plate, well_str):
    """Create labels/ subgroup inside a field with symlinks to label stores."""
    available_labels = []
    for label_type, plates in labels_data.items():
        if (
            plate in plates
            and well_str in plates[plate]
            and tile in plates[plate][well_str]
        ):
            available_labels.append((label_type, Path(plates[plate][well_str][tile])))

    if not available_labels:
        return

    labels_dir = field_dir / "labels"
    _write_labels_group_metadata(labels_dir, [name for name, _ in available_labels])

    for label_name, label_store in available_labels:
        label_dir = labels_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)

        # Copy label store's zarr.json
        shutil.copy2(label_store / "zarr.json", label_dir / "zarr.json")

        # Symlink each pyramid level
        _symlink_pyramid_levels(label_store, label_dir)


def _nest_image_subgroup(field_dir, image_type, source_store):
    """Create a named subgroup inside a field for an additional image type."""
    subgroup_dir = field_dir / image_type
    subgroup_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_store / "zarr.json", subgroup_dir / "zarr.json")
    _symlink_pyramid_levels(source_store, subgroup_dir)


def _write_labels_group_metadata(labels_dir, label_names):
    """Write labels group zarr.json listing available labels."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "labels": label_names,
            }
        },
    }
    with open(labels_dir / "zarr.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
            "ome": {
                "version": "0.5",
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
                },
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
            "ome": {
                "version": "0.5",
                "well": {
                    "images": [
                        {"path": str(idx), "acquisition": 0} for idx in field_indices
                    ],
                },
            }
        },
    }

    with open(well_dir / "zarr.json", "w") as f:
        json.dump(well_metadata, f, indent=2)
