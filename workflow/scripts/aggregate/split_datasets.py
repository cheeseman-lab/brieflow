import pandas as pd
import numpy as np

from lib.aggregate.cell_classification import CellClassifier
from lib.aggregate.cell_data_utils import (
    load_metadata_cols,
    split_cell_data,
    channel_combo_subset,
)


def apply_confidence_thresholds(metadata, features, thresholds, class_col="class"):
    """Apply per-class confidence thresholds with mode support.

    Args:
        metadata: DataFrame with classification results (must have class_col and 'confidence')
        features: DataFrame with feature data (aligned with metadata)
        thresholds: Either a scalar threshold (legacy) or dict of per-class thresholds:
            {class_id: {"threshold": float, "mode": "exclude"|"reassign"}, ...}
        class_col: Name of the column containing class predictions

    Returns:
        Filtered (metadata, features) tuple
    """
    if thresholds is None:
        return metadata, features

    before_count = len(metadata)

    # Handle legacy scalar threshold format
    if isinstance(thresholds, (int, float)):
        mask = metadata["confidence"] >= thresholds
        metadata = metadata[mask]
        features = features[mask]
        print(f"Filtered by confidence >= {thresholds}: {before_count} -> {len(metadata)} cells")
        return metadata, features

    # Handle new per-class threshold format
    if not isinstance(thresholds, dict):
        print(f"Warning: Unrecognized threshold format {type(thresholds)}, skipping filtering")
        return metadata, features

    # Build mask for cells to keep
    keep_mask = np.ones(len(metadata), dtype=bool)

    for class_id, config in thresholds.items():
        # Handle both int and string keys (YAML may parse as either)
        class_id = int(class_id) if isinstance(class_id, str) else class_id

        if isinstance(config, (int, float)):
            # Simple threshold value
            threshold = config
            mode = "exclude"
        elif isinstance(config, dict):
            threshold = config.get("threshold", 0.5)
            mode = config.get("mode", "exclude")
        else:
            print(f"Warning: Unrecognized config format for class {class_id}, skipping")
            continue

        # Find cells of this class
        class_mask = metadata[class_col] == class_id
        class_confidence = metadata.loc[class_mask, "confidence"]

        # Apply threshold based on mode
        if mode == "exclude":
            # Drop cells below threshold
            low_conf_mask = class_mask & (metadata["confidence"] < threshold)
            keep_mask = keep_mask & ~low_conf_mask
            dropped = low_conf_mask.sum()
            print(f"Class {class_id}: excluded {dropped} cells below threshold {threshold}")

        elif mode == "reassign":
            # For reassign mode, we'd need to re-predict with the classifier
            # For now, treat as exclude (reassign requires classifier access)
            low_conf_mask = class_mask & (metadata["confidence"] < threshold)
            keep_mask = keep_mask & ~low_conf_mask
            dropped = low_conf_mask.sum()
            print(f"Class {class_id}: excluded {dropped} cells below threshold {threshold} (reassign mode not yet implemented in pipeline)")

        else:
            print(f"Warning: Unknown mode '{mode}' for class {class_id}, using exclude")
            low_conf_mask = class_mask & (metadata["confidence"] < threshold)
            keep_mask = keep_mask & ~low_conf_mask

    metadata = metadata[keep_mask]
    features = features[keep_mask]
    print(f"Total after thresholding: {before_count} -> {len(metadata)} cells")

    return metadata, features


# Load merge data
cell_data = pd.read_parquet(snakemake.input[0])

# Split data into metadata and features
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp)
metadata, features = split_cell_data(cell_data, metadata_cols)

# Classify cells only if classifier path is provided
classifier_path = snakemake.params.get("classifier_path")
confidence_thresholds = snakemake.params.get("confidence_thresholds")

if classifier_path is not None:
    print("Applying cell classification...")
    classifier = CellClassifier.load(classifier_path)
    metadata, features = classifier.classify_cells(metadata, features)

    # Apply confidence thresholds (supports both legacy scalar and new per-class format)
    metadata, features = apply_confidence_thresholds(
        metadata, features, confidence_thresholds, class_col="class"
    )
else:
    print("No classifier specified - skipping cell classification")

# Load all channels
all_channels = snakemake.params.all_channels

# split cells by cell class
for cell_class in snakemake.params.cell_classes:
    if cell_class == "all":
        cell_class_metadata = metadata
        cell_class_features = features
    else:
        cell_class_mask = metadata["class"] == cell_class
        cell_class_metadata = metadata[cell_class_mask]
        cell_class_features = features[cell_class_mask]

    # split features into channel combos
    for channel_combo in snakemake.params.channel_combos:
        channel_combo_list = channel_combo.split("_")
        channel_combo_features = channel_combo_subset(
            cell_class_features, channel_combo_list, all_channels
        )

        # concatenate metadata and features
        cell_class_data = pd.concat(
            [cell_class_metadata, channel_combo_features], axis=1
        ).reset_index(drop=True)

        # Save data
        dataset_fp = [
            f
            for f in snakemake.output
            if f"CeCl-{cell_class}_ChCo-{channel_combo}__" in f
        ][0]
        cell_class_data.to_parquet(dataset_fp, index=False)
