"""Write HCS plate-level OME-NGFF metadata for plate zarr directories.

This script discovers the structure of plate zarr directories that were
populated directly by Snakemake jobs and writes the necessary zarr.json
metadata files at plate, row, well, and labels levels.
"""

from pathlib import Path

from lib.shared.hcs import write_hcs_metadata

from iohub.ngff import open_ome_zarr
import re
import pandas as pd

def _parse_plate_from_store_name(store_path: Path) -> str:
    """
    aligned_1.zarr -> "1"
    illumination_corrected_12.zarr -> "12"
    """
    m = re.search(r"_(\d+)\.zarr$", store_path.name)
    if not m:
        raise ValueError(f"Could not parse plate from store name: {store_path.name}")
    return m.group(1)


def _infer_modality_from_store_path(store_path: Path) -> str:
    # expects .../sbs/<store>.zarr or .../phenotype/<store>.zarr
    parts = store_path.parts
    if "sbs" in parts:
        return "sbs"
    if "phenotype" in parts:
        return "phenotype"
    raise ValueError(f"Could not infer modality from path: {store_path}")


def _load_pixel_size_map(
    preprocess_root: Path, modality: str, plate: str
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """
    Returns (row, col, tile) -> (px_x, px_y) in micrometers.
    Reads preprocess/metadata/{modality}/{plate}/{row}/{col}/combined_metadata.parquet
    """
    meta_root = preprocess_root / "metadata" / modality / plate
    pixel_map: dict[tuple[str, str, str], tuple[float, float]] = {}

    if not meta_root.exists():
        print(f"[iohub patch] metadata root missing: {meta_root}")
        return pixel_map

    for fp in meta_root.rglob("combined_metadata.parquet"):
        rel = fp.relative_to(meta_root)
        if len(rel.parts) < 3:
            continue
        row, col = str(rel.parts[0]), str(rel.parts[1])
        df = pd.read_parquet(fp)

        if "tile" not in df.columns:
            # fallback: constant per well
            if "pixel_size_x" in df.columns and "pixel_size_y" in df.columns and len(df) > 0:
                pixel_map[(row, col, "*")] = (float(df["pixel_size_x"].iloc[0]), float(df["pixel_size_y"].iloc[0]))
            continue

        # per-tile mapping
        for _, r in df.iterrows():
            tile = str(r["tile"])
            if pd.isna(r.get("pixel_size_x")) or pd.isna(r.get("pixel_size_y")):
                continue
            pixel_map[(row, col, tile)] = (float(r["pixel_size_x"]), float(r["pixel_size_y"]))

    return pixel_map


def _ensure_axes_units_micrometer(pos) -> None:
    """
    Ensures spatial axes (x, y, z) have unit="micrometer" on the iohub
    metadata model.  Must modify pos.metadata (not pos.zattrs) so that
    changes survive pos.dump_meta().
    """
    if not pos.metadata.multiscales:
        return
    for ax in pos.metadata.multiscales[0].axes:
        if ax.name.lower() in ("x", "y", "z"):
            ax.unit = "micrometer"


def _maybe_rename_channels(pos, channel_names: list[str]) -> None:
    """
    Renames channels if the current channel list matches c0,c1,... pattern.
    Only renames up to min(n_channels, len(channel_names)).
    """
    try:
        current = list(pos.channel_names)
    except Exception:
        return

    n = min(len(current), len(channel_names))
    for i in range(n):
        old = current[i]
        new = channel_names[i]
        if old != new:
            try:
                pos.rename_channel(old, new)
            except Exception:
                # fallback: do nothing if rename fails
                pass
    
def patch_store_metadata_with_iohub(store_path: Path, preprocess_root: Path, channel_names: list[str] | None = None):
    """
    Opens store in r+ and patches:
      - x/y scale using combined_metadata.parquet
      - axis units to micrometer
      - optional channel renames
    """
    plate = _parse_plate_from_store_name(store_path)
    modality = _infer_modality_from_store_path(store_path)
    pixel_map = _load_pixel_size_map(preprocess_root, modality, plate)

    print(f"[patch] store={store_path}")
    print(f"[patch] pixel_map entries={len(pixel_map)}")

    ds = open_ome_zarr(str(store_path), layout="hcs", mode="r+", version="0.5")
    pos_list = list(ds.positions())
    print(f"[patch] positions={len(pos_list)}, first={pos_list[0][0] if pos_list else None}")

    for pos_path, pos in pos_list:
        parts = pos_path.split("/")
        if len(parts) != 3:
            print(f"[patch] skipping unexpected pos_path={pos_path}")
            continue

        row, col, tile = parts
        key = (str(row), str(col), str(tile))
        fallback = (str(row), str(col), "*")

        if key in pixel_map:
            px_x, px_y = pixel_map[key]
        elif fallback in pixel_map:
            px_x, px_y = pixel_map[fallback]
        else:
            px_x = px_y = None

        print(f"[patch] pos={pos_path} key={key} px=({px_x}, {px_y})")

        if px_x is not None:
            pos.set_scale("*", "x", px_x)
            pos.set_scale("*", "y", px_y)

        if channel_names:
            _maybe_rename_channels(pos, channel_names)

        _ensure_axes_units_micrometer(pos)
        pos.dump_meta()

    ds.dump_meta()
    ds.close()


plate_zarr_dirs = snakemake.params.plate_zarr_dirs
channels_metadata = getattr(snakemake.params, "channels_metadata", None)

# robust preprocess_root (based on config root_fp)
root_fp = Path(snakemake.config["all"]["root_fp"])
preprocess_root = root_fp / "preprocess"


total = 0
for plate_zarr in plate_zarr_dirs:
    plate_path = Path(plate_zarr)
    if plate_path.exists():
        print(f"Writing HCS metadata for: {plate_path}")
        write_hcs_metadata(plate_path, channels_metadata=channels_metadata)
        total += 1
        # decide channel_names for this store
        channel_names = getattr(snakemake.params, "channel_names", None)
        # skip preprocess stores for iohub patching (preprocess has extra cycle level)
        if "preprocess" not in plate_path.parts:
            patch_store_metadata_with_iohub(
                plate_path,
                preprocess_root,
                channel_names=channel_names,
            )
    else:
        print(f"Plate zarr not found, skipping: {plate_path}")

if total > 0:
    print(f"\nHCS metadata written for {total} plate zarr(s).")
else:
    print("No plate zarr directories found. Skipping HCS metadata.")

