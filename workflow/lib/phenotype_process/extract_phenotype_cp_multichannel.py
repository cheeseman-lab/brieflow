from itertools import combinations, permutations, product

import pandas as pd

from lib.phenotype_process.cp_emulator import (
    grayscale_features_multichannel,
    correlation_features_multichannel,
    shape_features,
    grayscale_columns_multichannel,
    correlation_columns_multichannel,
    shape_columns,
    neighbor_measurements,
)
from lib.shared.extract_phenotype_minimal import extract_features


def extract_phenotype_cp_multichannel(
    data_phenotype,
    nuclei,
    cells,
    wildcards,
    cytoplasms=None,
    nucleus_channels="all",
    cell_channels="all",
    cytoplasm_channels="all",
    foci_channel=None,
    channel_names=["dapi", "tubulin", "gh2ax", "phalloidin"],
):
    """Extract phenotype features from CellProfiler-like data with multi-channel functionality.

    Parameters:
    - data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
    - nuclei (numpy.ndarray): Nuclei segmentation data.
    - cells (numpy.ndarray): Cell segmentation data.
    - cytoplasms (numpy.ndarray, optional): Cytoplasmic segmentation data.
    - wildcards (dict): Dictionary containing wildcards.
    - nucleus_channels (str or list): List of nucleus channel indices to consider or 'all'.
    - cell_channels (str or list): List of cell channel indices to consider or 'all'.
    - foci_channel (int): Index of the channel containing foci information.
    - channel_names (list): List of channel names.

    Returns:
    - pandas.DataFrame: DataFrame containing extracted phenotype features.
    """
    # Check if all channels should be used
    if nucleus_channels == "all":
        try:
            nucleus_channels = list(range(data_phenotype.shape[-3]))
        except:
            nucleus_channels = [0]

    if cell_channels == "all":
        try:
            cell_channels = list(range(data_phenotype.shape[-3]))
        except:
            cell_channels = [0]

    if cytoplasm_channels == "all":
        try:
            cytoplasm_channels = list(range(data_phenotype.shape[-3]))
        except:
            cytoplasm_channels = [0]

    dfs = []

    # Define features
    features = grayscale_features_multichannel
    features.update(correlation_features_multichannel)
    features.update(shape_features)

    # Define function to create column map
    def make_column_map(channels):
        columns = {}
        # Create columns for grayscale features
        for feat, out in grayscale_columns_multichannel.items():
            columns.update(
                {
                    f"{feat}_{n}": f"{channel_names[ch]}_{renamed}"
                    for n, (renamed, ch) in enumerate(product(out, channels))
                }
            )
        # Create columns for correlation features
        for feat, out in correlation_columns_multichannel.items():
            if feat == "lstsq_slope":
                iterator = permutations
            else:
                iterator = combinations
            columns.update(
                {
                    f"{feat}_{n}": renamed.format(
                        first=channel_names[first], second=channel_names[second]
                    )
                    for n, (renamed, (first, second)) in enumerate(
                        product(out, iterator(channels, 2))
                    )
                }
            )
        # Add shape columns
        columns.update(shape_columns)
        return columns

    # Create column maps for nucleus and cell
    nucleus_columns = make_column_map(nucleus_channels)
    cell_columns = make_column_map(cell_channels)

    # Extract nucleus features
    dfs.append(
        extract_features(
            data_phenotype[..., nucleus_channels, :, :],
            nuclei,
            wildcards,
            features,
            multichannel=True,
        )
        .rename(columns=nucleus_columns)
        .set_index("label")
        .rename(columns=lambda x: "nucleus_" + x if x not in wildcards.keys() else x)
    )

    # Extract cell features
    dfs.append(
        extract_features(
            data_phenotype[..., cell_channels, :, :],
            cells,
            dict(),
            features,
            multichannel=True,
        )
        .rename(columns=cell_columns)
        .set_index("label")
        .add_prefix("cell_")
    )

    # Extract cytoplasmic features if cytoplasms are provided
    if cytoplasms is not None:
        cytoplasmic_columns = make_column_map(cytoplasm_channels)
        dfs.append(
            extract_features(
                data_phenotype[..., cytoplasm_channels, :, :],
                cytoplasms,
                dict(),
                features,
                multichannel=True,
            )
            .rename(columns=cytoplasmic_columns)
            .set_index("label")
            .add_prefix("cytoplasm_")
        )

    # Extract foci features if foci channel is provided
    if foci_channel is not None:
        foci = ops.process.find_foci(
            data_phenotype[..., foci_channel, :, :], remove_border_foci=True
        )
        dfs.append(
            Snake_ph._extract_features_bare(foci, cells, features=ops.features.foci)
            .set_index("label")
            .add_prefix(f"cell_{channel_names[foci_channel]}_")
        )

    # Extract nucleus and cell neighbors
    dfs.append(
        neighbor_measurements(nuclei, distances=[1])
        .set_index("label")
        .add_prefix("nucleus_")
    )

    dfs.append(
        neighbor_measurements(cells, distances=[1])
        .set_index("label")
        .add_prefix("cell_")
    )
    if cytoplasms is not None:
        dfs.append(
            neighbor_measurements(cytoplasms, distances=[1])
            .set_index("label")
            .add_prefix("cytoplasm_")
        )

    # Concatenate data frames and reset index
    return pd.concat(dfs, axis=1, join="outer", sort=True).reset_index()
