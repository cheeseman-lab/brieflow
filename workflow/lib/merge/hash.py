"""Utility functions for hashing cells in merge module."""

import warnings

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor, LinearRegression


def hash_process_info(process_info_df):
    """Generate hashed Delaunay triangulation for process info at screen level.

    1) Preprocess table of `i, j` coordinates (typically nuclear centroids) to ensure at least 4 valid cells per tile.
    2) Computes a Delaunay triangulation of the input points.

    Args:
        process_info_df (pandas.DataFrame): Table of points with columns `i`, `j`, and tile of cell.

    Returns:
        pandas.DataFrame: Table containing a hashed Delaunay triangulation, with one row per simplex (triangle).
    """
    # Ensure that i and j are not null, at least 4 cells per tile
    process_info_df = process_info_df[
        process_info_df["i"].notnull() & process_info_df["j"].notnull()
    ]
    process_info_df = process_info_df.groupby(["well", "tile"]).filter(
        lambda x: len(x) > 3
    )

    # Group the DataFrame by "tile" and apply find_triangles sequentially
    process_info_hash = []

    for tile, group in process_info_df.groupby("tile"):
        result = find_triangles(group)
        result = result.assign(tile=tile)  # Add the tile column back to the result
        process_info_hash.append(result)

    # Concatenate all the results into a single DataFrame
    process_info_hash = pd.concat(process_info_hash, ignore_index=True)

    return process_info_hash


def find_triangles(df):
    """Generates a hashed Delaunay triangulation for input points.

    Processes a table of `i, j` coordinates (typically nuclear centroids) and computes a Delaunay triangulation of the input points. Each tile/site is processed independently. The triangulations for all tiles/sites within a single well are concatenated and used as input to `multistep_alignment`.

    Args:
        df (pandas.DataFrame): Table of points with columns `i` and `j`.

    Returns:
        pandas.DataFrame: Table containing a hashed Delaunay triangulation, with one row per simplex (triangle).
    """
    # Extract the coordinates from the dataframe and compute the Delaunay triangulation
    v, c = get_vectors(df[["i", "j"]].values)

    # Create a dataframe from the vectors and rename the columns with a prefix 'V_'
    df_vectors = pd.DataFrame(v).rename(columns="V_{0}".format)

    # Create a dataframe from the coordinates and rename the columns with a prefix 'c_'
    df_coords = pd.DataFrame(c).rename(columns="c_{0}".format)

    # Concatenate the two dataframes along the columns
    df_combined = pd.concat([df_vectors, df_coords], axis=1)

    # Assign a new column 'magnitude' which is the Euclidean distance (magnitude) of each vector
    df_result = df_combined.assign(magnitude=lambda x: x.eval("(V_0**2 + V_1**2)**0.5"))

    return df_result


def get_vectors(X):
    """Calculates edge vectors and centers for all faces in the Delaunay triangulation.

    Computes the nine edge vectors and centers for each face in the Delaunay triangulation of the given point array `X`.

    Args:
        X (numpy.ndarray): Array of points to be triangulated.

    Returns:
        tuple:
            - numpy.ndarray: Array of shape (n_faces, 18) containing the vector displacements for the nine edges of each triangle.
            - numpy.ndarray: Array of shape (n_faces, 2) containing the center points of each triangle.
    """
    dt = Delaunay(X)  # Create Delaunay triangulation of the points
    vectors, centers = [], []  # Initialize lists to store vectors and centers

    for i in range(dt.simplices.shape[0]):
        # Skip triangles with an edge on the outer boundary
        if (dt.neighbors[i] == -1).any():
            continue

        result = nine_edge_hash(
            dt, i
        )  # Get the nine edge vectors for the current triangle
        # Some rare event where hashing fails
        if result is None:
            continue

        _, v = result  # Unpack the result to get the vectors
        c = X[dt.simplices[i], :].mean(axis=0)  # Calculate the center of the triangle
        vectors.append(v)  # Append the vectors to the list
        centers.append(c)  # Append the center to the list

    # Convert lists to numpy arrays and reshape vectors to (n_faces, 18)
    return np.array(vectors).reshape(-1, 18), np.array(centers)


def nine_edge_hash(dt, i):
    """Extracts vector displacements for edges connected to a specified triangle in a Delaunay triangulation.

    For triangle `i` in the Delaunay triangulation `dt`, computes the vector displacements for the 9 edges containing at least one vertex of the triangle. Raises an error if the triangle lies on the outer boundary of the triangulation.

    Example:
        ```python
        dt = Delaunay(X_0)
        i = 0
        segments, vector = nine_edge_hash(dt, i)
        ```

    Args:
        dt (scipy.spatial.Delaunay): Delaunay triangulation object containing points and simplices.
        i (int): Index of the triangle in the Delaunay triangulation.

    Returns:
        tuple:
            - list[tuple]: List of vertex pairs representing the 9 edges.
            - numpy.ndarray: Array containing vector displacements for the 9 edges.
    """
    # Indices of inner three vertices in CCW order
    a, b, c = dt.simplices[i]

    # Reorder vertices so that the edge 'ab' is the longest
    X = dt.points
    start = np.argmax((np.diff(X[[a, b, c, a]], axis=0) ** 2).sum(axis=1) ** 0.5)
    if start == 0:
        order = [0, 1, 2]
    elif start == 1:
        order = [1, 2, 0]
    elif start == 2:
        order = [2, 0, 1]
    a, b, c = np.array([a, b, c])[order]

    # Get indices of outer three vertices connected to the inner vertices
    a_ix, b_ix, c_ix = dt.neighbors[i]
    inner = {a, b, c}
    outer = lambda xs: [x for x in xs if x not in inner][0]

    try:
        bc = outer(dt.simplices[dt.neighbors[i, order[0]]])
        ac = outer(dt.simplices[dt.neighbors[i, order[1]]])
        ab = outer(dt.simplices[dt.neighbors[i, order[2]]])
    except IndexError:
        return None

    if any(x == -1 for x in (bc, ac, ab)):
        error = "triangle on outer boundary, neighbors are: {0} {1} {2}"
        raise ValueError(error.format(bc, ac, ab))

    # Define the 9 edges
    segments = [
        (a, b),
        (b, c),
        (c, a),
        (a, ab),
        (b, ab),
        (b, bc),
        (c, bc),
        (c, ac),
        (a, ac),
    ]

    # Extract the vector displacements for the 9 edges
    i_coords = X[segments, 0]
    j_coords = X[segments, 1]
    vector = np.hstack([np.diff(i_coords, axis=1), np.diff(j_coords, axis=1)])

    return segments, vector


def initial_alignment(df_0, df_1, initial_sites=8):
    """Identifies matching tiles from two acquisitions with similar Delaunay triangulations within the same well.

    Matches tiles from two datasets based on Delaunay triangulations, assuming minimal cell movement between acquisitions and equivalent segmentations.

    Args:
        df_0 (pandas.DataFrame): Hashed Delaunay triangulation for all tiles in dataset 0. Produced by concatenating outputs of `find_triangles` for individual tiles of a single well. Must include a `tile` column.
        df_1 (pandas.DataFrame): Hashed Delaunay triangulation for all sites in dataset 1. Produced by concatenating outputs of `find_triangles` for individual sites of a single well. Must include a `site` column.
        initial_sites (int | list[tuple[int, int]], optional): If an integer, specifies the number of sites sampled from `df_1` for initial brute-force matching of tiles to build the alignment model. If a list of 2-tuples, represents known (tile, site) matches to initialize the alignment model. At least 5 pairs are recommended.

    Returns:
        pandas.DataFrame: Table of possible (tile, site) matches, including rotation and translation transformations. Includes all tested matches, which should be filtered by `score` and `determinant` to retain valid matches.
    """

    # Define a function to work on individual (tile,site) pairs
    def work_on(df_t, df_s):
        rotation, translation, score = evaluate_match(df_t, df_s)
        determinant = None if rotation is None else np.linalg.det(rotation)
        result = pd.Series(
            {
                "rotation": rotation,
                "translation": translation,
                "score": score,
                "determinant": determinant,
            }
        )
        return result

    arr = []
    for tile, site in initial_sites:
        result = work_on(df_0.query("tile==@tile"), df_1.query("site==@site"))
        result.at["site"] = site
        result.at["tile"] = tile
        arr.append(result)
    df_initial = pd.DataFrame(arr)

    return df_initial


def evaluate_match(df_0, df_1, threshold_triangle=0.3, threshold_point=2):
    """Evaluates the match between two sets of vectors and centers.

    Computes the transformation parameters (rotation and translation) and evaluates the quality of the match between two datasets based on their vectors and centers.

    Args:
        df_0 (pandas.DataFrame): DataFrame containing the first set of vectors and centers.
        df_1 (pandas.DataFrame): DataFrame containing the second set of vectors and centers.
        threshold_triangle (float, optional): Threshold for matching triangles. Defaults to 0.3.
        threshold_point (float, optional): Threshold for matching points. Defaults to 2.

    Returns:
        tuple:
            - numpy.ndarray: Rotation matrix of the transformation.
            - numpy.ndarray: Translation vector of the transformation.
            - float: Score of the transformation based on the matching points.
    """
    V_0, c_0 = get_vc(df_0)  # Extract vectors and centers from the first DataFrame
    V_1, c_1 = get_vc(df_1)  # Extract vectors and centers from the second DataFrame

    i0, i1, distances = nearest_neighbors(
        V_0, V_1
    )  # Find nearest neighbors between the vectors

    # Filter triangles based on distance threshold
    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]  # Get the matching centers

    # Minimum number of matching triangles required to proceed
    if sum(filt) < 5:
        return None, None, -1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Use matching triangles to define transformation
        model = RANSACRegressor()
        model.fit(X, Y)  # Fit the RANSAC model to the matching centers

    rotation = model.estimator_.coef_  # Extract rotation matrix
    translation = model.estimator_.intercept_  # Extract translation vector

    # Score transformation based on the triangle centers
    distances = cdist(model.predict(c_0), c_1, metric="sqeuclidean")
    threshold_region = 50  # Threshold for the region to consider
    filt = np.sqrt(distances.min(axis=0)) < threshold_region
    score = (
        np.sqrt(distances.min(axis=0))[filt] < threshold_point
    ).mean()  # Calculate score

    return rotation, translation, score  # Return rotation, translation, and score


def get_vc(df, normalize=True):
    """Extracts vectors and centers from the DataFrame, with optional normalization of vectors.

    Args:
        df (pandas.DataFrame): DataFrame containing vectors and centers.
        normalize (bool, optional): Whether to normalize the vectors. Defaults to True.

    Returns:
        tuple:
            - numpy.ndarray: Array of vectors.
            - numpy.ndarray: Array of centers.
    """
    V, c = (
        df.filter(like="V").values,
        df.filter(like="c").values,
    )  # Extract vectors and centers
    if normalize:
        V = (
            V / df["magnitude"].values[:, None]
        )  # Normalize the vectors by their magnitudes
    return V, c  # Return vectors and centers


def nearest_neighbors(V_0, V_1):
    """Computes the nearest neighbors between two sets of vectors.

    Args:
        V_0 (numpy.ndarray): First set of vectors.
        V_1 (numpy.ndarray): Second set of vectors.

    Returns:
        tuple:
            - numpy.ndarray: Indices of the nearest neighbors in `V_0`.
            - numpy.ndarray: Indices of the nearest neighbors in `V_1`.
            - numpy.ndarray: Distances between the nearest neighbors.
    """
    Y = cdist(V_0, V_1, metric="sqeuclidean")  # Compute squared Euclidean distances
    distances = np.sqrt(
        Y.min(axis=1)
    )  # Compute the smallest distances and take the square root
    ix_0 = np.arange(V_0.shape[0])  # Indices of V_0
    ix_1 = Y.argmin(axis=1)  # Indices of nearest neighbors in V_1
    return ix_0, ix_1, distances  # Return indices and distances


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
