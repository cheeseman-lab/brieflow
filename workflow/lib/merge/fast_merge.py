"""Functions for merging SBS and phenotype cell information across wells."""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def merge_triangle_hash(
    hash_df_0, hash_df_1, alignment, threshold=2, local_refinement=None, warp_kwargs=None
):
    """Merges two DataFrames using triangle hashing after images at different magnifications have been hashed together.

    Args:
        hash_df_0 (pandas.DataFrame): The first DataFrame.
        hash_df_1 (pandas.DataFrame): The second DataFrame.
        alignment (dict): Alignment parameters containing rotation and translation.
        threshold (int): The threshold value. Defaults to 2.
        local_refinement (str | bool | None): If truthy, refine the global affine with a
            local non-rigid warp before matching, to correct residual within-tile
            distortion (e.g. two-scope acquisitions). The string selects the warp model:
            "polynomial" or "thin_plate_spline". Defaults None (off) — behaviour is then
            identical to the plain affine merge.
        warp_kwargs (dict | None): Keyword args forwarded to `refine_local_warp`
            (degree, iterations, min_correspondences, smoothing, max_correspondences) when
            local_refinement is on. Defaults None.

    Returns:
        pandas.DataFrame: The merged DataFrame.
    """
    # Rename 'tile' column to 'site' in hash_df_1
    hash_df_1 = hash_df_1.rename(columns={"tile": "site"})

    # Build linear model
    model = build_linear_model(alignment["rotation"], alignment["translation"])

    # Merge dataframes using triangle hashing
    return merge_sbs_phenotype(
        hash_df_0,
        hash_df_1,
        model,
        threshold=threshold,
        local_refinement=local_refinement,
        warp_kwargs=warp_kwargs,
    )


def refine_local_warp(
    X, Y, Y_pred, threshold, degree=2, iterations=2, min_correspondences=30,
    model="polynomial", smoothing=10.0, max_correspondences=700
):
    """Refine a global affine alignment with a local non-rigid warp.

    Corrects residual within-tile distortion that a single affine cannot capture (the
    failure mode when SBS and phenotype come from differently-configured microscopes).
    The warp is fit ONLY on high-confidence correspondences — points already matched
    within `threshold` under the current prediction — so it cannot be pulled by spurious
    loose matches. Each iteration the warp improves and more points come into range.

    Two warp models (``model``):
      - "polynomial" (default): a degree-``degree`` polynomial. Backward-compatible;
        the original behaviour. Beware high degrees — a degree-5 polynomial oscillates
        (Runge) and can diverge.
      - "thin_plate_spline": a smoothed thin-plate spline (scipy RBFInterpolator). It is
        the smoothest deformation fitting the correspondences, so it captures local
        distortion without the polynomial's oscillation, regularized by ``smoothing``.
        Validated to beat the polynomial on two-scope OPS merges (higher single-match
        rate, ~half the median residual) while remaining stable.

    Degrades gracefully: with fewer than `min_correspondences` confident matches it
    returns the input prediction unchanged (so sparse tiles behave like plain affine).

    Args:
        X (numpy.ndarray): Source (phenotype) coordinates, shape (n, 2).
        Y (numpy.ndarray): Target (SBS) coordinates, shape (m, 2).
        Y_pred (numpy.ndarray): Current predicted source coords in target space (n, 2),
            i.e. the global-affine prediction.
        threshold (float): Match distance defining a high-confidence correspondence.
        degree (int): Polynomial degree (model="polynomial"). Defaults to 2.
        iterations (int): Number of refine-and-rematch passes. Defaults to 2.
        min_correspondences (int): Minimum confident matches required to fit. Defaults 30.
        model (str): "polynomial" (default) or "thin_plate_spline".
        smoothing (float): TPS regularization (model="thin_plate_spline"). Defaults 10.0.
        max_correspondences (int): TPS fit is capped at this many points (subsampled) for
            speed; the fit is smooth so a few hundred anchors suffice. Defaults 700.

    Returns:
        numpy.ndarray: Refined predicted source coordinates in target space, shape (n, 2).
    """
    pf = PolynomialFeatures(degree)
    refined = Y_pred
    for _ in range(iterations):
        distances = cdist(Y, refined, metric="sqeuclidean")
        nearest = distances.argmin(axis=1)
        within = np.sqrt(distances.min(axis=1)) < threshold
        if within.sum() < min_correspondences:
            break
        Y_corr = Y[within]
        if model == "thin_plate_spline":
            from scipy.interpolate import RBFInterpolator

            # TPS is fit on the RESIDUAL (current prediction -> target), not raw
            # source -> target: its `smoothing` is calibrated for the small residual
            # scale, so fitting the full transform (which carries the ~0.27x scale)
            # would mis-regularize. Compose the correction onto the running estimate.
            src = refined[nearest[within]]
            if len(src) > max_correspondences:
                sel = np.random.default_rng(0).choice(
                    len(src), max_correspondences, replace=False
                )
                src, Y_corr = src[sel], Y_corr[sel]
            rbf = RBFInterpolator(
                src, Y_corr, kernel="thin_plate_spline", smoothing=smoothing
            )
            refined = rbf(refined)
        else:
            X_corr = X[nearest[within]]
            reg = LinearRegression().fit(pf.fit_transform(X_corr), Y_corr)
            refined = reg.predict(pf.transform(X))
    return refined


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


def merge_sbs_phenotype(
    cell_locations_0,
    cell_locations_1,
    model,
    threshold=2,
    local_refinement=None,
    warp_kwargs=None,
):
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
        local_refinement (str | bool | None, optional): If truthy, refine the affine
            prediction with a local warp (`refine_local_warp`) before matching. The string
            "polynomial" or "thin_plate_spline" selects the model. Defaults None (off) —
            prediction is then the plain affine, identical to before.
        warp_kwargs (dict | None, optional): Keyword args forwarded to `refine_local_warp`
            (degree, iterations, min_correspondences, smoothing, max_correspondences) when
            local_refinement is on. Defaults None.

    Returns:
        pandas.DataFrame: Table of merged identities of cell labels from cell_locations_0 and cell_locations_1.
        Returns empty DataFrame with correct columns if input is empty.
    """
    # Final columns for the merged DataFrame
    cols_final = [
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

    # Optional local non-rigid refinement of the affine prediction (corrects residual
    # within-tile distortion; off by default so existing screens are unaffected).
    # `local_refinement` doubles as the warp-model selector: the string "polynomial" or
    # "thin_plate_spline" picks the model, while a bare truthy (e.g. True) keeps
    # refine_local_warp's default (polynomial). An explicit warp_kwargs["model"] wins.
    if local_refinement:
        wk = dict(warp_kwargs or {})
        if isinstance(local_refinement, str):
            wk.setdefault("model", local_refinement)
        Y_pred = refine_local_warp(X, Y, Y_pred, threshold, **wk)

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
