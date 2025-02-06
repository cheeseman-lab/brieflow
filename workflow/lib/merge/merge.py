"""Functions for merging SBS and phenotype cell information across wells."""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression


def merge_triangle_hash(hash_df_0, hash_df_1, alignment, threshold=2):
    """Merges two DataFrames using triangle hashing after images at different magnifications have been hashed together.

    Args:
        hash_df_0 (pandas.DataFrame): The first DataFrame.
        hash_df_1 (pandas.DataFrame): The second DataFrame.
        alignment (dict): Alignment parameters containing rotation and translation.
        threshold (int): The threshold value. Defaults to 2.

    Returns:
        pandas.DataFrame: The merged DataFrame.
    """
    # Rename 'tile' column to 'site' in hash_df_1
    hash_df_1 = hash_df_1.rename(columns={"tile": "site"})

    # Build linear model
    model = build_linear_model(alignment["rotation"], alignment["translation"])

    # Merge dataframes using triangle hashing
    return merge_sbs_phenotype(hash_df_0, hash_df_1, model, threshold=threshold)


def build_linear_model(rotation, translation):
    """Builds a linear regression model using the provided rotation matrix and translation vector.

    Args:
        rotation (numpy.ndarray): Rotation matrix for the model.
        translation (numpy.ndarray): Translation vector for the model.

    Returns:
        sklearn.linear_model.LinearRegression: Linear regression model with the specified rotation
        and translation.
    """
    m = LinearRegression()
    m.coef_ = rotation  # Set the rotation matrix as the model's coefficients
    m.intercept_ = translation  # Set the translation vector as the model's intercept
    return m  # Return the linear regression model


def merge_sbs_phenotype(cell_locations_0, cell_locations_1, model, threshold=2):
    """Perform fine alignment of one (tile, site) match found using `multistep_alignment`.

    Args:
        cell_locations_0 (pandas.DataFrame): Table of coordinates to align (e.g., nuclei centroids)
            for one tile of dataset 0. Expects `i` and `j` columns.
        cell_locations_1 (pandas.DataFrame): Table of coordinates to align (e.g., nuclei centroids)
            for one site of dataset 1 that was identified as a match
            to the tile in cell_locations_0 using `multistep_alignment`. Expects
            `i` and `j` columns.
        model (sklearn.linear_model.LinearRegression): Linear alignment model suggested
            to be passed in, functioning between tile of cell_locations_0 and site of cell_locations_1.
            Produced using `build_linear_model` with the rotation and translation
            matrix determined in `multistep_alignment`.
        threshold (float, optional): Maximum Euclidean distance allowed between matching
            points. Defaults to 2.

    Returns:
        pandas.DataFrame: Table of merged identities of cell labels from cell_locations_0 and cell_locations_1.
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
        cell_locations_0 is None
        or cell_locations_1 is None
        or (hasattr(cell_locations_0, "empty") and cell_locations_0.empty)
        or (hasattr(cell_locations_1, "empty") and cell_locations_1.empty)
    ):
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=cols_final)

    # Extract coordinates from the DataFrames
    X = cell_locations_0[["i", "j"]].values  # Coordinates from dataset 0
    Y = cell_locations_1[["i", "j"]].values  # Coordinates from dataset 1

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
    target = (
        cell_locations_0.iloc[ix[filt]].reset_index(drop=True).rename(columns=columns_0)
    )

    # Merge DataFrames and calculate distances
    return (
        cell_locations_1[filt]
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
