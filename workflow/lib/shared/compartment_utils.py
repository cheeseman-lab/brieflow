"""Helpers for optional compartment-specific pipeline outputs."""

from pathlib import Path

import pandas as pd


def get_default_compartment_combo(second_obj_detection: bool) -> str:
    """Return the all-compartments combo for the configured phenotype run.

    Args:
        second_obj_detection: Whether secondary-object detection is enabled.

    Returns:
        The hyphen-delimited all-compartments combo.
    """
    compartments = ["cell", "nucleus", "cytoplasm"]
    if second_obj_detection:
        compartments.append("second_obj")
    return "-".join(compartments)


def normalize_compartment_combo_table(
    combos: pd.DataFrame,
    split_by_compartment: bool,
    default_compartment_combo: str,
) -> pd.DataFrame:
    """Normalize a wildcard-combo table for the configured path mode.

    Args:
        combos: Aggregate or cluster wildcard combinations.
        split_by_compartment: Whether compartment-specific paths are enabled.
        default_compartment_combo: Combo used when paths are not compartment-specific
            or the input table omits the column.

    Returns:
        A normalized copy of ``combos`` with a ``compartment_combo`` column.
    """
    normalized = combos.copy()
    if not split_by_compartment or "compartment_combo" not in normalized.columns:
        normalized["compartment_combo"] = default_compartment_combo
    return normalized.drop_duplicates().reset_index(drop=True)


def normalize_compartment_records(
    records: list[dict],
    split_by_compartment: bool,
    default_compartment_combo: str,
) -> list[dict]:
    """Normalize compartment values in config records and remove duplicates.

    Args:
        records: Configuration records, such as bootstrap combinations.
        split_by_compartment: Whether compartment-specific paths are enabled.
        default_compartment_combo: Combo used when paths are not compartment-specific
            or a record omits the value.

    Returns:
        Normalized records in their original order.
    """
    normalized_records = []
    for record in records:
        normalized = dict(record)
        if not split_by_compartment or not normalized.get("compartment_combo"):
            normalized["compartment_combo"] = default_compartment_combo
        if normalized not in normalized_records:
            normalized_records.append(normalized)
    return normalized_records


def get_compartment_combo(
    wildcards,
    split_by_compartment: bool,
    default_compartment_combo: str,
) -> str:
    """Resolve a compartment combo without requiring an off-mode wildcard.

    Args:
        wildcards: Snakemake wildcards for the current job.
        split_by_compartment: Whether compartment-specific paths are enabled.
        default_compartment_combo: Combo to use when splitting is disabled.

    Returns:
        The wildcard value when splitting is enabled, otherwise the default combo.
    """
    if split_by_compartment:
        return wildcards.compartment_combo
    return default_compartment_combo


def add_compartment_metadata(
    metadata: dict,
    compartment_combo: str,
    split_by_compartment: bool,
) -> dict:
    """Add filename metadata only for compartment-specific paths.

    Args:
        metadata: Existing filename metadata.
        compartment_combo: Compartment value or wildcard template.
        split_by_compartment: Whether compartment-specific paths are enabled.

    Returns:
        A copy of ``metadata`` with optional compartment metadata.
    """
    resolved = dict(metadata)
    if split_by_compartment:
        resolved["compartment_combo"] = compartment_combo
    return resolved


def add_compartment_path(
    base_path: str | Path,
    compartment_combo: str,
    split_by_compartment: bool,
) -> Path:
    """Append a compartment directory only when splitting is enabled.

    Args:
        base_path: Path before the optional compartment segment.
        compartment_combo: Compartment value or wildcard template.
        split_by_compartment: Whether compartment-specific paths are enabled.

    Returns:
        The path with the optional compartment directory.
    """
    base_path = Path(base_path)
    if split_by_compartment:
        return base_path / compartment_combo
    return base_path


def add_compartment_suffix(
    stem: str,
    compartment_combo: str,
    split_by_compartment: bool,
) -> str:
    """Append a delimiter-safe compartment suffix to a bootstrap stem.

    Args:
        stem: Bootstrap path stem before the optional compartment value.
        compartment_combo: Compartment value or wildcard template.
        split_by_compartment: Whether compartment-specific paths are enabled.

    Returns:
        ``stem`` with ``__{compartment_combo}`` appended only when enabled.
    """
    if split_by_compartment:
        return f"{stem}__{compartment_combo}"
    return stem
