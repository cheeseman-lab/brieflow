"""Bootstrap statistical testing for perturbation effects.

This module provides functionality for performing bootstrap statistical tests
to validate perturbation effects detected in the aggregate pipeline. It includes
functions for data preparation, feature selection, bootstrap simulation, and
p-value calculation at both construct and gene levels.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def get_construct_features(
    construct_id: str, construct_features_arr: np.ndarray
) -> np.ndarray:
    """Extract feature array for a specific construct.

    Args:
        construct_id: Identifier for the construct.
        construct_features_arr: Array containing all construct features.

    Returns:
        Feature array for the specified construct (excluding ID column).
    """
    indices = np.where(construct_features_arr[:, 0] == construct_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Construct {construct_id} not found in features array")
    construct_arr = construct_features_arr[indices][0][1:].astype(float)
    return construct_arr


def run_single_bootstrap_simulation(
    controls_arr: np.ndarray, sample_size: int
) -> np.ndarray:
    """Run a single bootstrap simulation.

    Args:
        controls_arr: Array of control data.
        sample_size: Size of sample to draw.

    Returns:
        Median values from the simulation.
    """
    random_cell_idx = np.random.choice(controls_arr.shape[0], 1, replace=False)[0]
    control_id = controls_arr[random_cell_idx, 0]

    indices = np.where(controls_arr[:, 0] == control_id)[0]
    control_construct_arr = controls_arr[indices]

    # Sample cells with replacement from this construct only
    sample_arr = control_construct_arr[
        np.random.choice(control_construct_arr.shape[0], sample_size, replace=True)
    ]

    # Return median (excluding ID column)
    return np.median(sample_arr[:, 1:].astype(float), axis=0)


def calculate_pvals(
    null_medians_arr: np.ndarray, observed_medians: np.ndarray
) -> np.ndarray:
    """Calculate two-tailed p-values from bootstrap null distribution.

    Args:
        null_medians_arr: Array of null distribution medians from simulations.
        observed_medians: Array of observed median values.

    Returns:
        Array of two-tailed p-values.
    """
    # Calculate one-tailed p-values
    pvals_one_tail = (null_medians_arr > observed_medians).mean(axis=0)

    # Convert to two-tailed by taking minimum and doubling
    pvals_two_tail = np.array([1 - x if x > 0.5 else x for x in pvals_one_tail])
    pvals_two_tail = pvals_two_tail * 2

    return pvals_two_tail


def run_construct_bootstrap(
    construct_id: str,
    construct_features_arr: np.ndarray,
    controls_arr: np.ndarray,
    sample_size: int,
    num_sims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run bootstrap analysis for a single construct.

    Args:
        construct_id: Identifier for the construct to analyze.
        construct_features_arr: Array containing all construct features.
        controls_arr: Array of control data.
        sample_size: Sample size for bootstrap simulations.
        num_sims: Number of bootstrap simulations to run.

    Returns:
        Tuple containing:
        - null_medians_arr: Array of null distribution medians
        - p_vals: Array of p-values for each feature
    """
    # Get observed medians for this construct
    observed_medians = get_construct_features(construct_id, construct_features_arr)

    # Initialize null distribution array
    null_medians_arr = np.zeros((num_sims, controls_arr.shape[1] - 1))

    # Run bootstrap simulations
    for i in range(num_sims):
        median_arr = run_single_bootstrap_simulation(controls_arr, sample_size)
        null_medians_arr[i, :] = median_arr

    # Modified verification for single feature case
    if null_medians_arr.shape[1] == 1:
        # For single feature, check if all values are identical (which would be weird)
        all_identical = len(np.unique(null_medians_arr)) == 1
        if all_identical:
            raise ValueError("All bootstrap simulations returned identical values")
    else:
        # Original verification for multiple features
        all_same = np.max(null_medians_arr, axis=1) == np.min(null_medians_arr, axis=1)
        if len(null_medians_arr[all_same]) > 0:
            raise ValueError(
                "Bootstrap simulation failed - some simulations returned constant values"
            )

    # Calculate p-values
    p_vals = calculate_pvals(null_medians_arr, observed_medians)

    return null_medians_arr, p_vals


def load_construct_null_arrays(file_paths: List[str]) -> List[np.ndarray]:
    """Load multiple construct null distribution arrays.

    Args:
        file_paths: List of paths to null distribution files.

    Returns:
        List of loaded numpy arrays.
    """
    return [np.load(path, allow_pickle=False) for path in file_paths]
