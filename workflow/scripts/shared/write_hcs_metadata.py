"""Assemble plate-level HCS OME-NGFF zarr stores from per-tile zarr outputs."""

from pathlib import Path

from lib.shared.hcs import assemble_hcs_plate, discover_zarr_stores

images_dir = Path(snakemake.params.images_dir)
hcs_dir = Path(snakemake.params.hcs_dir)


def _has_zarr_stores(path):
    """Check if a directory (or its children) directly contains .zarr stores."""
    return any(path.rglob("*.zarr"))


def _detect_modality_subdirs(images_path):
    """Detect modality subdirs (sbs/, phenotype/) vs direct plate layout.

    If the immediate children of images_path are NOT plate-like (numeric)
    but instead contain further subdirectories with zarr stores, they are
    modality subdirectories.  Returns a list of (modality_name, path) tuples,
    or [(None, images_path)] when the layout is direct.
    """
    subdirs = sorted(
        d for d in images_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not subdirs:
        return [(None, images_path)]

    # If the first subdir contains .zarr stores directly at the expected depth
    # (plate/well/tile/image.zarr), it's a direct layout.  Otherwise check if
    # it looks like a modality prefix (non-numeric name with zarr stores deeper).
    first = subdirs[0]
    if first.name.isdigit():
        # Numeric first child → plate directory → direct layout
        return [(None, images_path)]

    # Non-numeric children with zarr stores → modality subdirs
    modalities = []
    for d in subdirs:
        if _has_zarr_stores(d):
            modalities.append((d.name, d))
    return modalities if modalities else [(None, images_path)]


def _assemble_for_dir(sub_images_dir, sub_hcs_dir):
    """Discover zarr stores and assemble HCS plate zarrs for one directory."""
    images, labels = discover_zarr_stores(str(sub_images_dir))

    if not images:
        print(f"  No zarr stores found in {sub_images_dir}. Skipping.")
        return 0

    if labels:
        print(f"  Found label types: {sorted(labels.keys())}")
    print(f"  Found image types: {sorted(images.keys())}")

    all_plates = set()
    for plates_data in images.values():
        all_plates.update(plates_data.keys())

    for plate in sorted(all_plates):
        plate_images = {}
        for img_type, plates_data in images.items():
            if plate in plates_data:
                plate_images[img_type] = plates_data[plate]

        n_types = len(plate_images)
        n_wells = max(len(w) for w in plate_images.values())
        print(
            f"  Assembling HCS: {plate}.zarr ({n_types} image types, {n_wells} wells)"
        )
        assemble_hcs_plate(
            plate,
            plate_images,
            str(sub_hcs_dir),
            str(sub_images_dir),
            labels_data=labels,
        )

    return len(all_plates)


print(f"Discovering zarr stores in: {images_dir}")
modalities = _detect_modality_subdirs(images_dir)
total_plates = 0

for modality_name, modality_path in modalities:
    if modality_name:
        print(f"\nProcessing modality: {modality_name}")
        sub_hcs = hcs_dir / modality_name
    else:
        sub_hcs = hcs_dir
    total_plates += _assemble_for_dir(modality_path, sub_hcs)

if total_plates > 0:
    print(f"\nHCS assembly complete: {total_plates} plate stores created in {hcs_dir}")
else:
    print("No zarr stores found. Skipping HCS metadata generation.")
