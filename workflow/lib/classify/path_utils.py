"""Utilities for resolving parquet file paths in classification workflows."""

from pathlib import Path
from typing import Tuple, List, Optional

from lib.shared.file_utils import get_filename


def get_parquet_config(
    mode: str,
    source: str,
    root_fp: Path
) -> Tuple[Path, str]:
    """Get parquet directory and name suffix based on mode and source.

    Args:
        mode: "cell" or "vacuole"
        source: "phenotype" or "merge"
        root_fp: Root file path from config

    Returns:
        Tuple of (parquet_dir, name_suffix)

    Raises:
        ValueError: If mode or source is invalid
    """
    mode = str(mode).lower()
    source = str(source).lower()

    if mode not in {"cell", "vacuole"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'cell' or 'vacuole'.")

    if source not in {"phenotype", "merge"}:
        raise ValueError(f"Invalid source: {source}. Must be 'phenotype' or 'merge'.")

    if source == "merge":
        parquet_dir = root_fp / "merge" / "parquets"
        name_suffix = "merge_final"
    else:  # phenotype
        parquet_dir = root_fp / "phenotype" / "parquets"
        if mode == "cell":
            name_suffix = "phenotype_cp"
        else:  # vacuole
            name_suffix = "phenotype_vacuoles"

    return parquet_dir, name_suffix


def find_sample_parquet(
    plates: List[str],
    wells: List[str],
    parquet_dir: Path,
    name_suffix: str
) -> Optional[Path]:
    """Find first available parquet file from plate/well combinations.

    Args:
        plates: List of plate IDs as strings
        wells: List of well IDs as strings
        parquet_dir: Directory containing parquet files
        name_suffix: Filename suffix (e.g., "phenotype_cp")

    Returns:
        Path to first found parquet file, or None if none exist
    """
    for plate in sorted(plates):
        for well in sorted(wells):
            file_path = parquet_dir / get_filename(
                {"plate": int(plate), "well": well},
                name_suffix,
                "parquet"
            )
            if file_path.exists():
                return file_path

    return None
