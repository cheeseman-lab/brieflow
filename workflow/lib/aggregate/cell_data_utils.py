DEFAULT_METADATA_COLS = [
    "plate",
    "well",
    "tile",
    "cell_0",
    "i_0",
    "j_0",
    "site",
    "cell_1",
    "i_1",
    "j_1",
    "distance",
    "fov_distance_0",
    "fov_distance_1",
    "sgRNA_0",
    "gene_symbol_0",
    "mapped_single_gene",
    "channels_min",
    "nucleus_i",
    "nucleus_j",
    "nucleus_bounds_0",
    "nucleus_bounds_1",
    "nucleus_bounds_2",
    "nucleus_bounds_3",
    "cell_i",
    "cell_j",
    "cell_bounds_0",
    "cell_bounds_1",
    "cell_bounds_2",
    "cell_bounds_3",
    "cytoplasm_i",
    "cytoplasm_j",
    "cytoplasm_bounds_0",
    "cytoplasm_bounds_1",
    "cytoplasm_bounds_2",
    "cytoplasm_bounds_3",
]


def split_cell_data(cell_data, metadata_cols):
    """
    Splits the cell data into metadata and features.
    """

    metadata = cell_data[metadata_cols]
    features = cell_data.drop(columns=metadata_cols)
    return metadata, features


def channel_combo_subset(features, channel_combo):
    """Return subset of DataFrame with columns containing any of the given channel names."""
    cols = [col for col in features.columns if any(ch in col for ch in channel_combo)]
    return features[cols]
