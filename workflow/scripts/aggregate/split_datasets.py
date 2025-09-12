import pandas as pd

from lib.aggregate.cell_classification import CellClassifier
from lib.aggregate.cell_data_utils import (
    load_metadata_cols,
    split_cell_data,
    channel_combo_subset,
    compartment_subset,
)

# Load merge data
cell_data = pd.read_parquet(snakemake.input[0])

# Split data into metadata and features
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp)
metadata, features = split_cell_data(cell_data, metadata_cols)

# Classify cells
import numpy as np

# Classify cells only if classifier path is provided
classifier_path = snakemake.params.get("classifier_path")
if classifier_path is not None:
    print("Applying cell classification...")
    classifier = CellClassifier.load(classifier_path)
    metadata, features = classifier.classify_cells(metadata, features)
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

        # Apply compartment filtering after channel filtering
        for compartment_combo in snakemake.params.compartment_combos:
            compartment_combo_list = compartment_combo.split("_")
            compartment_filtered_features = compartment_subset(
                channel_combo_features, compartment_combo_list
            )

            # concatenate metadata and features
            cell_class_data = pd.concat(
                [cell_class_metadata, compartment_filtered_features], axis=1
            ).reset_index(drop=True)

            # Save data
            dataset_fp = [
                f
                for f in snakemake.output
                if cell_class in f and channel_combo in f and compartment_combo in f
            ][0]
            cell_class_data.to_parquet(dataset_fp, index=False)
