"""Utility functions for hashing cells in merge module.

These include:
- Preprocessing cell location dataframes for hashing.
- Generating hashed Delaunay triangulation for cells.
- Performing initial and multistep alignment.
"""

import warnings
import multiprocessing
from collections.abc import Iterable

import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor
from joblib import Parallel, delayed


def hash_cell_locations(cell_locations_df):
    """Generate hashed Delaunay triangulation for process info at screen level.

    1) Preprocess table of `i, j` coordinates (typically nuclear centroids) to ensure at least 4 valid cells per tile.
    2) Computes a Delaunay triangulation of the input points.

    Args:
        cell_locations_df (pandas.DataFrame): Table of points with columns `i`, `j`, and tile of cell.

    Returns:
        pandas.DataFrame: Table containing a hashed Delaunay triangulation, with one row per simplex (triangle).
    """
    # Ensure that i and j are not null, at least 4 cells per tile
    cell_locations_df = cell_locations_df[
        cell_locations_df["i"].notnull() & cell_locations_df["j"].notnull()
    ]
    cell_locations_df = cell_locations_df.groupby(["well", "tile"]).filter(
        lambda x: len(x) > 3
    )

    # Find triangles across well with parallel processing
    cell_locations_hash = cell_locations_df.pipe(
        gb_apply_parallel, ["tile"], find_triangles
    )

    return cell_locations_hash


def find_triangles(cell_locations_df):
    """Generates a hashed Delaunay triangulation for input points.

    Processes a table of `i, j` coordinates (typically nuclear centroids) and computes a Delaunay triangulation of the input points. Each tile/site is processed independently. The triangulations for all tiles/sites within a single well are concatenated and used as input to `multistep_alignment`.

    Args:
        cell_locations_df (pandas.DataFrame): Table of points with columns `i` and `j`.

    Returns:
        pandas.DataFrame: Table containing a hashed Delaunay triangulation, with one row per simplex (triangle).
    """
    # Extract the coordinates from the dataframe and compute the Delaunay triangulation
    v, c = get_vectors(cell_locations_df[["i", "j"]].values)

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


def initial_alignment(well_triangles_0, well_triangles_1, initial_sites=8):
    """Identifies matching tiles from two acquisitions with similar Delaunay triangulations within the same well.

    Matches tiles from two datasets based on Delaunay triangulations, assuming minimal cell movement between acquisitions and equivalent segmentations.

    Args:
        well_triangles_0 (pandas.DataFrame): Hashed Delaunay triangulation for all tiles in dataset 0. Produced by concatenating outputs of `find_triangles` for individual tiles of a single well. Must include a `tile` column.
        well_triangles_1 (pandas.DataFrame): Hashed Delaunay triangulation for all sites in dataset 1. Produced by concatenating outputs of `find_triangles` for individual sites of a single well. Must include a `site` column.
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
        result = work_on(
            well_triangles_0.query("tile==@tile"), well_triangles_1.query("site==@site")
        )
        result.at["site"] = site
        result.at["tile"] = tile
        arr.append(result)
    df_initial = pd.DataFrame(arr)

    return df_initial


def evaluate_match(
    vec_centers_0, vec_centers_1, threshold_triangle=0.3, threshold_point=2
):
    """Evaluates the match between two sets of vectors and centers.

    Computes the transformation parameters (rotation and translation) and evaluates the quality of the match between two datasets based on their vectors and centers.

    Args:
        vec_centers_0 (pandas.DataFrame): DataFrame containing the first set of vectors and centers.
        vec_centers_1 (pandas.DataFrame): DataFrame containing the second set of vectors and centers.
        threshold_triangle (float, optional): Threshold for matching triangles. Defaults to 0.3.
        threshold_point (float, optional): Threshold for matching points. Defaults to 2.

    Returns:
        tuple:
            - numpy.ndarray: Rotation matrix of the transformation.
            - numpy.ndarray: Translation vector of the transformation.
            - float: Score of the transformation based on the matching points.
    """
    V_0, c_0 = get_vc(
        vec_centers_0
    )  # Extract vectors and centers from the first DataFrame
    V_1, c_1 = get_vc(
        vec_centers_1
    )  # Extract vectors and centers from the second DataFrame

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


def get_vc(vec_centers, normalize=True):
    """Extracts vectors and centers from the DataFrame, with optional normalization of vectors.

    Args:
        vec_centers (pandas.DataFrame): DataFrame containing vectors and centers.
        normalize (bool, optional): Whether to normalize the vectors. Defaults to True.

    Returns:
        tuple:
            - numpy.ndarray: Array of vectors.
            - numpy.ndarray: Array of centers.
    """
    V, c = (
        vec_centers.filter(like="V").values,
        vec_centers.filter(like="c").values,
    )  # Extract vectors and centers
    if normalize:
        V = (
            V / vec_centers["magnitude"].values[:, None]
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


def multistep_alignment(
    well_triangles_0,
    well_triangles_1,
    well_locations_0,
    well_locations_1,
    det_range=(1.125, 1.186),
    score=0.1,
    initial_sites=8,
    batch_size=180,
    n_jobs=None,
):
    """Find tiles of two different acquisitions with matching Delaunay triangulations within the same well.

    Cells must not have moved significantly between acquisitions, and segmentations should be approximately equivalent.

    Args:
        well_triangles_0 (pandas.DataFrame): Hashed Delaunay triangulation for all tiles of dataset 0. Produced by
            concatenating outputs of `find_triangles` from individual tiles of a single well. Expects a `tile` column.
        well_triangles_1 (pandas.DataFrame): Hashed Delaunay triangulation for all sites of dataset 1. Produced by
            concatenating outputs of `find_triangles` from individual sites of a single well. Expects a `site` column.
        well_locations_0 (pandas.DataFrame): Table of global coordinates for each tile acquisition to match tiles
            of `well_triangles_0`. Expects `tile` as index and two columns of coordinates.
        well_locations_1 (pandas.DataFrame): Table of global coordinates for each site acquisition to match sites
            of `well_triangles_1`. Expects `site` as index and two columns of coordinates.
        det_range (tuple, optional): Range of acceptable values for the determinant of the rotation matrix
            when evaluating an alignment of a tile-site pair. The determinant measures scaling consistency
            within microscope acquisition settings. Defaults to (1.125, 1.186).
        score (float, optional): Threshold score value for filtering valid matches from spurious ones.
            Used for initial alignment. Defaults to 0.1.
        initial_sites (int | list[tuple], optional): If int, the number of sites to sample from `well_triangles_1` for initial
            brute force matching to build a global alignment model. If a list of 2-tuples, represents known
            (tile, site) matches to start building the model. Defaults to 8.
        batch_size (int, optional): Number of (tile, site) matches to evaluate per batch during global
            alignment model updates. Defaults to 180.
        n_jobs (int, optional): Number of parallel jobs to deploy using joblib. Defaults to None.

    Returns:
        pandas.DataFrame: Table of possible (tile, site) matches with corresponding rotation and translation
        transformations. All tested matches are included; query based on `score` and `determinant` to filter valid matches.
    """
    # If n_jobs is not provided, set it to one less than the number of CPU cores
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1

    # Define a function to work on individual (tile,site) pairs
    def work_on(tiles_df, sites_df):
        rotation, translation, score = evaluate_match(tiles_df, sites_df)
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

    # If initial_sites is provided as a list of known matches, use it directly
    if isinstance(initial_sites, list):
        arr = []
        for tile, site in initial_sites:
            result = work_on(
                well_triangles_0.query("tile==@tile"),
                well_triangles_1.query("site==@site"),
            )
            result.at["site"] = site
            result.at["tile"] = tile
            arr.append(result)
        df_initial = pd.DataFrame(arr)
    else:
        # Otherwise, sample initial_sites number of sites randomly from well_triangles_1
        sites = (
            pd.Series(well_locations_1.index)
            .sample(initial_sites, replace=False, random_state=0)
            .pipe(list)
        )
        # Use brute force to find initial pairs of matches between tiles and sites
        df_initial = brute_force_pairs(
            well_triangles_0, well_triangles_1.query("site == @sites"), n_jobs=n_jobs
        )

    # Unpack det_range tuple into d0 and d1
    d0, d1 = det_range

    # Define the gate condition for filtering matches based on determinant and score
    gate = "@d0 <= determinant <= @d1 & score > @score"

    # Initialize alignments list with the initial matches
    alignments = [df_initial.query(gate)]

    # Main loop for iterating until convergence
    while True:
        # Concatenate alignments and remove duplicates
        df_align = pd.concat(alignments, sort=True).drop_duplicates(["tile", "site"])

        # Extract tested and matched pairs
        tested = df_align.reset_index()[["tile", "site"]].values
        matches = df_align.query(gate).reset_index()[["tile", "site"]].values

        # Prioritize candidate pairs based on certain criteria
        candidates = prioritize(well_locations_0, well_locations_1, matches)
        candidates = remove_overlap(candidates, tested)

        print("matches so far: {0} / {1}".format(len(matches), df_align.shape[0]))

        # Prepare data for parallel processing
        work = []
        d_0 = dict(list(well_triangles_0.groupby("tile")))
        d_1 = dict(list(well_triangles_1.groupby("site")))
        for ix_0, ix_1 in candidates[:batch_size]:
            if ix_0 in d_0 and ix_1 in d_1:  # Only process if both keys exist
                work.append([d_0[ix_0], d_1[ix_1]])
            else:
                print(f"Skipping tile {ix_0}, site {ix_1} - not found in data")

        if not work:  # If no valid pairs found, end alignment
            print("No valid pairs to process")
            break

        # Perform parallel processing of work
        df_align_new = pd.concat(
            Parallel(n_jobs=n_jobs)(delayed(work_on)(*w) for w in work), axis=1
        ).T.assign(
            tile=[t for t, _ in candidates[: len(work)]],
            site=[s for _, s in candidates[: len(work)]],
        )

        # Append new alignments to the list
        alignments += [df_align_new]

        if len(df_align_new.query(gate)) == 0:
            break

    return df_align


def brute_force_pairs(tiles_df_0, tiles_df_1, n_jobs=-2):
    """Evaluate all pairs of tiles (sites) between two DataFrames to find the best matches.

    Args:
        tiles_df_0 (pandas.DataFrame): DataFrame containing the first set of tiles (sites).
        tiles_df_1 (pandas.DataFrame): DataFrame containing the second set of tiles (sites).
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -2, which
            uses all but one core.

    Returns:
        pandas.DataFrame: DataFrame containing the results of the matching process, sorted by score.
    """
    work = tiles_df_1.groupby("site")

    arr = []
    for site, df_s in work:

        def work_on(df_t):
            rotation, translation, score = evaluate_match(
                df_t, df_s
            )  # Evaluate match for each tile pair
            determinant = (
                None if rotation is None else np.linalg.det(rotation)
            )  # Calculate determinant if rotation is not None
            result = pd.Series(
                {
                    "rotation": rotation,
                    "translation": translation,
                    "score": score,
                    "determinant": determinant,
                }
            )  # Create a result series
            return result

        (
            tiles_df_0.pipe(
                gb_apply_parallel, "tile", work_on, n_jobs=n_jobs
            )  # Apply work_on function in parallel
            .assign(site=site)  # Assign site to the results
            .pipe(arr.append)  # Append results to the list
        )

    return (
        pd.concat(arr)
        .reset_index()  # Concatenate all results and reset index
        .sort_values("score", ascending=False)
    )  # Sort results by score in descending order


def gb_apply_parallel(df, cols, func, n_jobs=None, backend="loky"):
    """Apply a function to groups of a DataFrame in parallel.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (str or list): Column(s) to group by.
        func (callable): Function to apply to each group.
        n_jobs (int, optional): Number of parallel jobs. If None, uses (CPU count - 1). Defaults to None.
        backend (str, optional): Joblib parallel backend. Defaults to 'loky'.

    Returns:
        pd.DataFrame or pd.Series: Results of applying func to each group, combined into a single DataFrame or Series.
    """
    # Ensure cols is a list
    if isinstance(cols, str):
        cols = [cols]

    # Set number of jobs if not specified
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1

    # Group the DataFrame
    grouped = df.groupby(cols)
    names, work = zip(*grouped)

    # Apply function in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(w) for w in work)

    # Process results based on their type
    if isinstance(results[0], pd.DataFrame):
        # For DataFrame results
        arr = []
        for labels, df in zip(names, results):
            if not isinstance(labels, Iterable):
                labels = [labels]
            if df is not None:
                (df.assign(**{c: l for c, l in zip(cols, labels)}).pipe(arr.append))
        results = pd.concat(arr)
    elif isinstance(results[0], pd.Series):
        # For Series results
        if len(cols) == 1:
            results = pd.concat(results, axis=1).T.assign(**{cols[0]: names})
        else:
            labels = zip(*names)
            results = pd.concat(results, axis=1).T.assign(
                **{c: l for c, l in zip(cols, labels)}
            )
    elif isinstance(results[0], dict):
        # For dict results
        results = pd.DataFrame(results, index=pd.Index(names, name=cols)).reset_index()

    return results


def prioritize(well_locations_0, well_locations_1, matches):
    """Produce an Nx2 array of tile (site) identifiers predicted to match within a search radius based on existing matches.

    Args:
        well_locations_0 (pandas.DataFrame): DataFrame containing tile (site) information for the first set.
        well_locations_1 (pandas.DataFrame): DataFrame containing tile (site) information for the second set.
        matches (numpy.ndarray): Nx2 array of tile (site) identifiers representing existing matches.

    Returns:
        list of tuple: List of predicted matching tile (site) identifiers.
    """
    a = well_locations_0.loc[
        matches[:, 0]
    ].values  # Get coordinates of matching tiles from the first set
    b = well_locations_1.loc[
        matches[:, 1]
    ].values  # Get coordinates of matching tiles from the second set

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = RANSACRegressor()
        model.fit(a, b)  # Fit the RANSAC model to the matching coordinates

    # Predict coordinates for the first set and calculate distances to the second set
    predicted = model.predict(well_locations_0.values)
    distances = cdist(predicted, well_locations_1.values, metric="sqeuclidean")
    ix = np.argsort(distances.flatten())  # Sort distances to find the closest matches
    ix_0, ix_1 = np.unravel_index(
        ix, distances.shape
    )  # Get indices of the closest matches

    candidates = list(
        zip(well_locations_0.index[ix_0], well_locations_1.index[ix_1])
    )  # Create list of candidate matches

    return remove_overlap(candidates, matches)  # Remove overlapping matches


def remove_overlap(xs, ys):
    """Remove overlapping pairs from a list of candidates based on an existing set of matches.

    Args:
        xs (list of tuple): List of candidate pairs.
        ys (list of tuple): List of existing matches.

    Returns:
        list of tuple: List of candidate pairs with overlaps removed.
    """
    ys = set(
        map(tuple, ys)
    )  # Convert existing matches to a set of tuples for fast lookup
    return [
        tuple(x) for x in xs if tuple(x) not in ys
    ]  # Return candidates that are not in existing matches
