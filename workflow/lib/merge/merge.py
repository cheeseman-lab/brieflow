"""Functions for merging SBS and phenotype cell information across wells."""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def merge_sbs_phenotype(df_0_, df_1_, model, threshold=2):
    """Perform fine alignment of one (tile, site) match found using `multistep_alignment`.

    Args:
        df_0_ (pandas.DataFrame): Table of coordinates to align (e.g., nuclei centroids)
            for one tile of dataset 0. Expects `i` and `j` columns.
        df_1_ (pandas.DataFrame): Table of coordinates to align (e.g., nuclei centroids)
            for one site of dataset 1 that was identified as a match
            to the tile in df_0_ using `multistep_alignment`. Expects
            `i` and `j` columns.
        model (sklearn.linear_model.LinearRegression): Linear alignment model suggested
            to be passed in, functioning between tile of df_0_ and site of df_1_.
            Produced using `build_linear_model` with the rotation and translation
            matrix determined in `multistep_alignment`.
        threshold (float, optional): Maximum Euclidean distance allowed between matching
            points. Defaults to 2.

    Returns:
        pandas.DataFrame: Table of merged identities of cell labels from df_0_ and df_1_.
        Returns empty DataFrame with correct columns if input is empty.
    """
    # Final columns for the merged DataFrame
    cols_final = [
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
    ]

    # Check if either dataframe is None or empty
    if (
        df_0_ is None
        or df_1_ is None
        or (hasattr(df_0_, "empty") and df_0_.empty)
        or (hasattr(df_1_, "empty") and df_1_.empty)
    ):
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=cols_final)

    # Extract coordinates from the DataFrames
    X = df_0_[["i", "j"]].values  # Coordinates from dataset 0
    Y = df_1_[["i", "j"]].values  # Coordinates from dataset 1

    # Predict coordinates for dataset 0 using the alignment model
    Y_pred = model.predict(X)

    # Calculate squared Euclidean distances between predicted coordinates and dataset 1 coordinates
    distances = cdist(Y, Y_pred, metric="sqeuclidean")

    # Find the index of the nearest neighbor in Y_pred for each point in Y
    ix = distances.argmin(axis=1)

    # Filter matches based on the threshold distance
    filt = np.sqrt(distances.min(axis=1)) < threshold

    # Define new column names for merging
    columns_0 = {"tile": "tile", "cell": "cell_0", "i": "i_0", "j": "j_0"}
    columns_1 = {"site": "site", "cell": "cell_1", "i": "i_1", "j": "j_1"}

    # Prepare the target DataFrame with matched coordinates from dataset 0
    target = df_0_.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)

    # Merge DataFrames and calculate distances
    return (
        df_1_[filt]
        .reset_index(drop=True)[  # Filtered rows from dataset 1
            list(columns_1.keys())
        ]  # Select columns for dataset 1
        .rename(columns=columns_1)  # Rename columns for dataset 1
        .pipe(
            lambda x: pd.concat([target, x], axis=1)
        )  # Concatenate with target DataFrame
        .assign(distance=np.sqrt(distances.min(axis=1))[filt])[
            cols_final
        ]  # Assign distance column  # Select final columns
    )
