import pandas as pd

from lib.aggregate.cell_classification import CellClassifier
from lib.aggregate.cell_data_utils import (
    load_metadata_cols,
    split_cell_data,
    channel_combo_subset,
)

# Load merge data
cell_data = pd.read_parquet(snakemake.input[0])

# Split data into metadata and features
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp)
metadata, features = split_cell_data(cell_data, metadata_cols)

# Classify cells
import numpy as np

classifier = CellClassifier.load(snakemake.params.classifier_path)
metadata, features = classifier.classify_cells(metadata, features)

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
