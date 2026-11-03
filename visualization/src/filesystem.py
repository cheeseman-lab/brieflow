import glob
import logging
import os
import re
import sys
import pandas as pd
import streamlit as st
import zarr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from workflow.lib.shared.parquet_io import read_parquet

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".file_manifest"

# Zarr-mode outputs nest the data location as directories instead of encoding it in the
# filename (sbs/eval/mapping/1/mapping_overview.tsv vs P-1__mapping_overview.tsv).
# The trailing directory components below an anchor are the location chain, whose length
# tells us which keys it carries.
LOCATION_ANCHORS = ("eval", "tsvs", "parquets")
LOCATION_CHAIN_KEYS = {
    1: ("plate",),
    3: ("plate", "row", "col"),
    4: ("plate", "row", "col", "tile"),
}

# Module-level cache so the manifest is read at most once per process.
_manifest_cache: dict[str, list[str]] = {}


def _load_manifest(data_root: str) -> list[str] | None:
    """Load a pre-built file manifest if one exists at data_root/.file_manifest.

    The manifest is a newline-delimited list of paths relative to data_root.
    Generate it at build/deploy time with:
        find <data_root> -type f | sed 's|^<data_root>/||' > <data_root>/.file_manifest
    """
    if data_root in _manifest_cache:
        return _manifest_cache[data_root]

    manifest_path = os.path.join(data_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return None

    with open(manifest_path) as f:
        paths = [line.strip() for line in f if line.strip()]
    _manifest_cache[data_root] = paths
    logger.info("Loaded manifest with %d entries from %s", len(paths), manifest_path)
    return paths


def _find_data_root(root_dir: str) -> tuple[str, str] | None:
    """Walk up from root_dir looking for a .file_manifest.

    Returns (data_root, prefix) where prefix is the relative path from
    data_root to root_dir, or None if no manifest found.
    """
    candidate = os.path.normpath(root_dir)
    while True:
        if os.path.isfile(os.path.join(candidate, MANIFEST_FILENAME)):
            prefix = os.path.relpath(root_dir, candidate)
            if prefix == ".":
                prefix = ""
            return candidate, prefix
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent


def _zarr_store_of(rel_path):
    """Return the innermost ``.zarr`` ancestor of a path, or None if it is not inside a store."""
    parts = rel_path.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(".zarr"):
            return "/".join(parts[: i + 1])
    return None


def _apply_path_filters(files, include_any=None, include_all=None):
    """Apply include_any / include_all directory-component filters."""
    filtered = files
    if include_any and len(include_any) > 0:
        filtered = [
            f
            for f in filtered
            if any(item in os.path.normpath(f).split(os.sep) for item in include_any)
        ]
    if include_all and len(include_all) > 0:
        filtered = [
            f
            for f in filtered
            if all(item in os.path.normpath(f).split(os.sep) for item in include_all)
        ]
    return filtered


def read_table(path):
    """Read a tabular file as a DataFrame, dispatching on its extension.

    Parquet goes through the pipeline's own reader so the visualizer and the pipeline
    agree on dtypes; TSV/CSV keep the plain pandas path.
    """
    if str(path).endswith(".parquet"):
        return read_parquet(path)
    if str(path).endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def read_zarr_channel_names(store_path):
    """Read channel labels from an OME-Zarr store's omero metadata.

    Returns None when the store carries no channel metadata.
    """
    attrs = dict(zarr.open_group(str(store_path), mode="r").attrs)
    # OME-NGFF v0.5 nests omero under "ome"; v0.4 keeps it at the top level.
    omero = attrs.get("ome", {}).get("omero") or attrs.get("omero")
    if not omero:
        return None
    return [channel.get("label") for channel in omero.get("channels", [])]


def parse_nested_location(file_path):
    """Extract the data location a zarr-mode path encodes as directory levels.

    Returns a dict with some of ``plate``/``row``/``col``/``tile``, or an empty dict when
    the path carries no recognizable location chain (TIFF mode, where the location lives
    in the filename instead).
    """
    parts = os.path.normpath(file_path).split(os.sep)
    # A "__" in the basename marks a flat TIFF-mode filename, which carries its own location.
    if "__" in parts[-1]:
        return {}

    parts = parts[:-1]
    anchor = max(
        (i for i, part in enumerate(parts) if part in LOCATION_ANCHORS), default=None
    )
    if anchor is None:
        return {}

    chain = parts[anchor + 1 :]
    # An eval group directory ("mapping", "segmentation", ...) can sit between the anchor
    # and the location, so drop the leading component when the full chain has no valid shape.
    if len(chain) not in LOCATION_CHAIN_KEYS and chain:
        chain = chain[1:]
    if len(chain) not in LOCATION_CHAIN_KEYS:
        return {}

    return dict(zip(LOCATION_CHAIN_KEYS[len(chain)], chain))


class FileSystem:
    """
    Utility class for file system operations related to evaluation files.
    """

    @staticmethod
    @st.cache_data
    def find_files(root_dir, include_any=None, include_all=None, extensions=None):
        """
        Find all files with specified extensions in the directory tree with optional path filtering.

        If a .file_manifest file exists at or above root_dir, uses it instead of
        walking the filesystem. This avoids expensive os.walk/glob over slow
        filesystems like gcsfuse (245k+ files = 300k+ HTTP GCS API calls).

        Args:
            root_dir: The root directory to search in
            includes_any: List of strings where if the path includes any of the values, it is included
            include_all: List of strings where the path must include each and every element
            extensions: List of file extensions to search for (default: ['png', 'tsv']).
                Use 'zarr' to match OME-Zarr stores, which are directories rather than files.

        Returns:
            A list of file paths that match the filtering criteria
        """
        if extensions is None:
            extensions = ["png", "tsv"]

        ext_set = set(extensions)

        # Try manifest-based lookup (no filesystem walk needed)
        result = _find_data_root(root_dir)
        if result is not None:
            data_root, prefix = result
            manifest = _load_manifest(data_root)
            if manifest is not None:
                all_files = []
                seen_stores = set()
                for rel in manifest:
                    if prefix and not rel.startswith(prefix + "/"):
                        continue
                    # A zarr store is a directory, so the manifest only lists the chunk
                    # files inside it; collapse those back to the store they belong to.
                    store = _zarr_store_of(rel) if "zarr" in ext_set else None
                    if store is not None:
                        if store not in seen_stores:
                            seen_stores.add(store)
                            all_files.append(os.path.join(data_root, store))
                        continue
                    ext = rel.rsplit(".", 1)[-1] if "." in rel else ""
                    if ext in ext_set:
                        all_files.append(os.path.join(data_root, rel))
                return _apply_path_filters(all_files, include_any, include_all)

        # Fallback: glob the filesystem
        all_files = []
        for ext in extensions:
            files = glob.glob(f"{root_dir}/**/*.{ext}", recursive=True)
            all_files.extend(files)

        return _apply_path_filters(all_files, include_any, include_all)

    @staticmethod
    def extract_well_id(file_path):
        r"""
        Extract well identifier from a file path.

        Reads the TIFF-mode W-[A-Z]\d+ filename prefix, falling back to the row/col
        directories that zarr mode nests the location into.

        Args:
            file_path: Path to the file

        Returns:
            Well ID if found, None otherwise
        """
        match = re.search(r"W-([A-Z]\d+)", file_path)
        if match:
            return match.group(1)
        location = parse_nested_location(file_path)
        if "row" in location and "col" in location:
            return location["row"] + location["col"]
        return None

    @staticmethod
    def extract_plate_id(file_path):
        r"""
        Extract plate identifier from a file path.

        Reads the TIFF-mode P-\d+ filename prefix, falling back to the plate directory
        that zarr mode nests the location into.

        Args:
            file_path: Path to the file

        Returns:
            Plate ID if found, None otherwise
        """
        match = re.search(r"P-(\d+)", file_path)
        if match:
            return match.group(1)
        return parse_nested_location(file_path).get("plate")

    @staticmethod
    def extract_metric_name(file_path):
        """
        Extract metric name from the basename (the part after the last "__").

        Args:
            file_path: Path to the file

        Returns:
            Metric name if found, None otherwise
        """
        base = os.path.splitext(os.path.basename(file_path))[0]
        if "__" in base:
            return base.split("__")[-1]
        # Zarr-mode basenames are the bare metric name, with the location in the directories.
        return base or None

    @staticmethod
    def extract_leiden_resolution(file_path):
        r"""
        Extract Leiden resolution (format: LR-\d+) from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Leiden resolution if found, None otherwise
        """
        match = re.search(r"(LR-\d+)", file_path)
        return match.group(1) if match else None

    @staticmethod
    def extract_features(root_dir, files):
        """
        Extract features from PNG and TSV files including path information.

        Args:
            root_dir: Root directory to use as base for relative paths
            files: List of file paths to process
            omit_folders: Set of folder names to omit from directory levels (default: {'eval'})

        Returns:
            DataFrame containing extracted features and path information
        """
        features = []

        for file in files:
            # Convert to relative path
            rel_path = os.path.relpath(file, root_dir)
            dirname = os.path.dirname(rel_path)
            basename = os.path.basename(file)
            name, ext = os.path.splitext(basename)

            # Basic feature dictionary with new fields
            feature = {
                "file_path": rel_path,
                "dir": dirname,
                "basename": name,
                "ext": ext.lstrip("."),  # Remove the leading dot from extension
            }

            # Extract identifiers and metadata from file path
            feature["well_id"] = FileSystem.extract_well_id(rel_path)
            feature["plate_id"] = FileSystem.extract_plate_id(rel_path)
            feature["metric_name"] = FileSystem.extract_metric_name(rel_path)
            # feature['leiden_resolution'] = FileSystem.extract_leiden_resolution(rel_path)

            # Add directory levels, skipping omitted folders
            parts = dirname.split(os.sep)
            for i, part in enumerate(parts):
                feature[f"dir_level_{i}"] = part
            features.append(feature)

        return pd.DataFrame(features)
