"""Utility functions for handling cell data in the brieflow aggregation pipeline.

This module provides helper functions for manipulating cell data, including
loading metadata columns, splitting cell data into metadata and features,
and filtering features based on channel combinations.
"""

import pandas as pd

from lib.phenotype.constants import DEFAULT_METADATA_COLS

# joins a perturbation to its group values; not "__" (the filename separator parsed
# in lib/shared/rule_utils.py), not a glob metacharacter, not regex-special
GROUP_KEY_SEP = "="


def load_metadata_cols(metadata_cols_fp, include_classification_cols=False):
    """Load metadata column names from a file.

    Args:
        metadata_cols_fp (str): File path to the metadata columns list.
        include_classification_cols (bool, optional): Whether to include
            classification columns. Defaults to False.

    Returns:
        list: List of metadata column names.
    """
    metadata_cols = pd.read_csv(metadata_cols_fp, header=None, sep="\t")[0].tolist()

    if include_classification_cols:
        metadata_cols += [
            "class",
            "confidence",
            "cell_stage_confidence",
        ]

    return metadata_cols


def control_mask(perturbation_values, control_key, match="contains"):
    """Flag control perturbations, ignoring any group suffix on a composite key.

    Matching the whole composite would let a group value satisfy control_key, silently
    labelling perturbed cells as controls (e.g. control_key "DMSO" matching "MYC=DMSO").
    Stripping the suffix is a no-op when group_cols is unset.

    A list control_key matches exactly against any element, for libraries whose controls
    share no usable prefix (an ORF screen's EGFP_1 and H2B-EGFP_1). A string keeps the
    caller's historical convention, so existing screens are unaffected.

    Args:
        perturbation_values (pd.Series): Perturbation names, composite or plain.
        control_key (str | list): Control identifier, or a list of exact names.
        match (str, optional): How a string key matches, "contains" or "startswith".
            Ignored for a list. Defaults to "contains".

    Returns:
        pd.Series: Boolean mask of control rows.
    """
    perturbations = perturbation_values.astype(str).str.split(GROUP_KEY_SEP, n=1).str[0]

    if isinstance(control_key, (list, tuple, set)):
        keys = set(control_key)
        # prepare_alignment_data uniquifies controls to <name>_<pert_id>, so an exact
        # match alone would stop seeing them downstream of that rename
        renamed = perturbations.str.startswith(tuple(f"{key}_" for key in keys))

        return perturbations.isin(keys) | renamed
    if match == "startswith":
        return perturbations.str.startswith(control_key, na=False)

    return perturbations.str.contains(control_key, na=False)


def join_well_annotations(metadata, well_annotations_fp):
    """Join per-well annotations onto cell metadata for splitting or grouping.

    Annotations are experimental variables assigned by plate map rather than derived
    from images, e.g. treatment, confluency, or timepoint. Keyed on (plate, well) so a
    multi-plate screen can carry a different map per plate.

    Args:
        metadata (pd.DataFrame): Cell metadata containing plate and well columns.
        well_annotations_fp (str): Path to a TSV with plate, well, and annotation columns.

    Returns:
        pd.DataFrame: Metadata with the annotation columns added.

    Raises:
        ValueError: If the map repeats a (plate, well), or if a (plate, well) present in
            the data has no row in the map.
    """
    annotations = pd.read_csv(well_annotations_fp, sep="\t")
    for col in ("plate", "well"):
        if col not in annotations.columns:
            raise ValueError(
                f"{well_annotations_fp} is missing required column '{col}'"
            )

    # a repeated well would multiply cells through the join and desync them from features
    duplicated = annotations[
        annotations.duplicated(subset=["plate", "well"], keep=False)
    ]
    if len(duplicated) > 0:
        raise ValueError(
            f"{well_annotations_fp} repeats (plate, well): "
            f"{sorted(map(tuple, duplicated[['plate', 'well']].to_numpy()))}"
        )

    # join on strings so plate 1 and "1" cannot silently fail to match
    keys = ["plate", "well"]
    metadata = metadata.copy()
    for col in keys:
        metadata[col] = metadata[col].astype(str)
        annotations[col] = annotations[col].astype(str)

    data_wells = set(map(tuple, metadata[keys].drop_duplicates().to_numpy()))
    map_wells = set(map(tuple, annotations[keys].drop_duplicates().to_numpy()))

    # an unmapped well would become a NaN group that silently drops its cells
    unmapped = sorted(data_wells - map_wells)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} (plate, well) in the data are absent from "
            f"{well_annotations_fp}: {unmapped}"
        )

    unused = sorted(map_wells - data_wells)
    if unused:
        print(f"Note: {len(unused)} annotated wells are not in the data: {unused}")

    # merge returns a fresh RangeIndex; features still carry the original one
    joined = metadata.merge(annotations, on=keys, how="left")
    joined.index = metadata.index

    return joined


RESERVED_METADATA_PREFIXES = ("offset_",)
# per-cell segmentation records (secondary object labels, nuclei count); not features
RESERVED_METADATA_COLS = ("second_obj_ids", "num_nuclei")


def is_reserved_metadata_col(col):
    """Return True for columns always treated as metadata regardless of METADATA_COLS.

    Per-cell alignment-offset QC columns (offset_*) have screen-specific names (the
    alignment step/cycle count varies by screen), so they are matched by prefix rather
    than enumerated in the metadata_cols file. Per-cell records written by secondary
    object detection and by segmentation are reserved by name.
    """
    return col in RESERVED_METADATA_COLS or col.startswith(RESERVED_METADATA_PREFIXES)


def split_cell_data(
    cell_data, metadata_cols, validate_dtypes=True, raise_on_invalid=True
):
    """Splits the cell data into metadata and features.

    Args:
        cell_data (pd.DataFrame): Input DataFrame containing cell data.
        metadata_cols (list): List of column names that represent metadata.
        validate_dtypes (bool, optional): Whether to validate that feature columns
            have numeric dtypes. Defaults to True.
        raise_on_invalid (bool, optional): Whether to raise an error if invalid
            dtypes are found. If False, only prints a warning. Defaults to True.

    Returns:
        tuple: (metadata, features) where metadata is a DataFrame containing
            only metadata columns and features is a DataFrame containing all
            non-metadata columns.

    Raises:
        ValueError: If validate_dtypes=True, raise_on_invalid=True, and non-numeric
            feature columns are detected.
    """
    # Ensure all metadata columns exist in the data
    existing_metadata_cols = [col for col in metadata_cols if col in cell_data.columns]

    # Reserved metadata (offset_* QC) matched by pattern, not enumeration
    reserved_metadata_cols = [
        col
        for col in cell_data.columns
        if col not in existing_metadata_cols and is_reserved_metadata_col(col)
    ]
    existing_metadata_cols = existing_metadata_cols + reserved_metadata_cols

    # Get metadata columns
    metadata = cell_data[existing_metadata_cols].copy()

    # Get feature columns (all columns not in metadata)
    features = cell_data.drop(columns=existing_metadata_cols).copy()

    # Validate feature dtypes
    if validate_dtypes:
        print("Validating feature columns ...")
        invalid_cols = []
        for col in features.columns:
            dtype = features[col].dtype
            if dtype == "object" or dtype.name == "object":
                invalid_cols.append(col)

        if invalid_cols:
            error_msg = f"\nWARNING: Found {len(invalid_cols)} non-numeric columns in features!\n"
            error_msg += "These columns should be added to METADATA_COLS:\n\n"
            for col in invalid_cols:
                sample_val = features[col].iloc[0] if len(features) > 0 else "N/A"
                error_msg += (
                    f"  - {col}: dtype={features[col].dtype}, sample='{sample_val}'\n"
                )
            error_msg += f"\nAdd these to CANDIDATE_METADATA_COLS (or the metadata_cols parameter) to fix."

            if raise_on_invalid:
                raise ValueError(
                    f"Invalid feature dtypes detected: {invalid_cols}\n{error_msg}"
                )
            else:
                print(error_msg)
        else:
            print("All feature columns have valid numeric dtypes")

    return metadata, features


def channel_combo_subset(features, channel_combo, all_channels):
    """Filter features to include only columns from specified channel combination.

    Args:
        features (pd.DataFrame): DataFrame containing feature data.
        channel_combo (list): List of channels to include.
        all_channels (list): List of all available channels.

    Returns:
        pd.DataFrame: DataFrame with features filtered to include only
            columns from the specified channel combination.
    """
    # Find channels to remove (those not in channel_combo)
    channels_to_remove = [ch for ch in all_channels if ch not in channel_combo]

    # Get all column names
    columns = features.columns.tolist()

    # Find columns to remove (those containing removed channel names)
    columns_to_remove = [
        col for col in columns if any(ch in col for ch in channels_to_remove)
    ]

    # Keep all columns except those from removed channels
    columns_to_keep = [col for col in columns if col not in columns_to_remove]

    return features[columns_to_keep]


def get_feature_table_cols(feature_cols, extra_tags=None):
    """Filter feature columns based on specific tags and compartments.

    Args:
        feature_cols (list): List of feature column names.
        extra_tags (list, optional): Additional substring tags to include on top of
            the default nucleus/cell subset. Any column whose name contains one of
            these tags (e.g. "radial_cv", "frac_at_d") is added. Defaults to None.

    Returns:
        list: Filtered list of feature column names.
    """
    # Define the specific tags to look for
    intensity_tags = [
        "mean",
        "integrated",
        "mass_displacement",
        "mean_edge",
        "std_edge",
        "mean_frac_0",
        "mean_frac_3",
    ]
    shape_tags = ["area", "solidity", "form_factor", "eccentricity"]
    # Substring match — picks up *_correlation_* (Pearson/Spearman) and *manders* features
    overlap_tags = ["manders", "correlation"]
    extra_tags = [tag.lower() for tag in extra_tags] if extra_tags else []

    # Define the specific compartments to look for
    compartments = ["nucleus", "cell"]

    # Initialize lists to store columns for each feature type
    intensity_cols = []
    shape_cols = []
    overlap_cols = []
    extra_cols = []

    # Filter columns based on compartments and tags
    for col in feature_cols:
        # Only include columns for nucleus or cell compartments
        if any(compartment in col for compartment in compartments):
            # Intensity features - must be at END of string
            if any(col.lower().endswith(tag) for tag in intensity_tags):
                intensity_cols.append(col)

            # Shape features - must be at END of string
            elif any(col.lower().endswith(tag) for tag in shape_tags):
                shape_cols.append(col)

            # Overlap features - can be anywhere in string
            elif any(tag in col.lower() for tag in overlap_tags):
                overlap_cols.append(col)

            # Extra user-requested features - substring match anywhere in string
            elif any(tag in col.lower() for tag in extra_tags):
                extra_cols.append(col)

    # Create a new DataFrame with selected columns, preserving the label column if it exists
    selected_columns = []
    if "label" in feature_cols:
        selected_columns.append("label")

    # Add columns in an organized way with clear section breaks
    selected_columns.extend(intensity_cols)
    selected_columns.extend(shape_cols)
    selected_columns.extend(overlap_cols)
    selected_columns.extend(extra_cols)

    return selected_columns


COMPARTMENT_PREFIXES = {
    "cell": "cell_",
    "nucleus": "nucleus_",
    "cytoplasm": "cytoplasm_",
    "second_obj": "second_obj_",
}

# Second-object-derived per-cell summary columns lacking the second_obj_ prefix.
SECOND_OBJ_EXTRA_COLS = frozenset(
    {
        "total_second_obj_area",
        "mean_second_obj_diameter",
        "mean_distance_to_nucleus",
    }
)


def compartment_combo_subset(
    features: pd.DataFrame, compartment_combo: str, all_compartments: list[str]
) -> pd.DataFrame:
    """Filter features to keep only columns belonging to the requested compartments.

    Args:
        features (pd.DataFrame): Feature columns (metadata already split out).
        compartment_combo (list[str]): Compartments to keep, e.g. ["cell", "nucleus"].
        all_compartments (list[str]): Compartments present in the input. Used to
            determine which prefixes to exclude.

    Returns:
        pd.DataFrame: features without columns belonging to excluded compartments.
    """
    excluded = [c for c in all_compartments if c not in compartment_combo]
    excluded_prefixes = [COMPARTMENT_PREFIXES[c] for c in excluded]

    cols_to_drop = [
        col
        for col in features.columns
        if any(col.startswith(p) for p in excluded_prefixes)
    ]
    if "second_obj" in excluded:
        cols_to_drop += [c for c in SECOND_OBJ_EXTRA_COLS if c in features.columns]

    return features.drop(columns=cols_to_drop)


def resolve_aggregate_combos(
    aggregate_combos: list[dict], second_obj_detection: bool
) -> list[dict]:
    """Normalize and validate AGGREGATE_COMBOS entries.

    For each entry:
    - Fills in default compartments (all 4 if detection on, 3 otherwise) when missing.
    - Dedupes compartments within the combo (preserving order).
    - Validates compartment names, non-empty channels/compartments, and
      second_obj-vs-detection consistency.
    After normalization, duplicate (channels, compartments) pairs are deduped.

    Args:
        aggregate_combos (list[dict]): Each dict has "channels" (list[str]) and
            optionally "compartments" (list[str]).
        second_obj_detection (bool): From config["phenotype"]["second_obj_detection"].

    Returns:
        list[dict]: Normalized, validated, and de-duplicated combos.

    Raises:
        ValueError: On any validation failure.
    """
    valid_compartments = set(COMPARTMENT_PREFIXES)
    default_compartments = (
        ["cell", "nucleus", "cytoplasm", "second_obj"]
        if second_obj_detection
        else ["cell", "nucleus", "cytoplasm"]
    )

    resolved = []
    seen = set()
    for idx, combo in enumerate(aggregate_combos):
        channels = list(combo.get("channels") or [])
        if not channels:
            raise ValueError(
                f"AGGREGATE_COMBOS[{idx}] must specify at least one channel"
            )

        comps = combo.get("compartments")
        if comps is None:
            comps = list(default_compartments)
        else:
            comps = list(comps)
        if not comps:
            raise ValueError(
                f"AGGREGATE_COMBOS[{idx}] must specify at least one compartment"
            )

        # Dedupe compartments while preserving order
        deduped = []
        for c in comps:
            if c not in valid_compartments:
                raise ValueError(
                    f"AGGREGATE_COMBOS[{idx}] has unknown compartment {c!r}; "
                    f"must be one of {sorted(valid_compartments)}"
                )
            if c == "second_obj" and not second_obj_detection:
                raise ValueError(
                    f"AGGREGATE_COMBOS[{idx}] lists 'second_obj' but "
                    f"config['phenotype']['second_obj_detection'] is False"
                )
            if c not in deduped:
                deduped.append(c)

        key = (tuple(channels), tuple(deduped))
        if key in seen:
            continue
        seen.add(key)
        resolved.append({"channels": channels, "compartments": deduped})

    return resolved
