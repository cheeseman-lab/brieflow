"""Bootstrap statistical testing for perturbation effects.

This module provides functionality for performing bootstrap statistical tests
to validate perturbation effects detected in the aggregate pipeline. It includes
functions for data preparation, feature selection, bootstrap simulation, and
p-value calculation at both construct and gene levels.
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Tuple, List
from scipy.stats import false_discovery_control


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


def create_pseudogene_groups(
    construct_features_df, pseudogene_patterns, perturbation_col
):
    """Group constructs into pseudo-genes based on patterns."""
    if not pseudogene_patterns:
        return [], construct_features_df

    print(f"Pseudogene patterns provided: {pseudogene_patterns}")

    pseudogene_groups = []
    remaining_constructs = construct_features_df.copy()

    for category_name, config in pseudogene_patterns.items():
        pattern = config["pattern"]
        constructs_per_pseudogene = config["constructs_per_pseudogene"]

        print(f"Processing category '{category_name}' with pattern '{pattern}'")

        # Find constructs matching this pattern
        mask = remaining_constructs[perturbation_col].str.match(pattern, na=False)
        matching_constructs = remaining_constructs[mask]

        if len(matching_constructs) == 0:
            print(
                f"  No constructs found matching pattern '{pattern}' for category '{category_name}'"
            )
            continue

        print(
            f"  Found {len(matching_constructs)} constructs matching pattern '{pattern}'"
        )
        print(f"  Examples: {matching_constructs[perturbation_col].head(3).tolist()}")

        # Remove matched constructs from remaining pool
        remaining_constructs = remaining_constructs[~mask]

        # Shuffle and group into pseudo-genes
        construct_list = matching_constructs.to_dict("records")
        random.shuffle(construct_list)

        # Create pseudo-gene groups
        pseudogene_counter = 1
        for i in range(0, len(construct_list), constructs_per_pseudogene):
            group = construct_list[i : i + constructs_per_pseudogene]
            pseudogene_id = f"{category_name}_pseudogene_{pseudogene_counter:02d}"

            pseudogene_groups.append(
                {
                    "pseudogene_id": pseudogene_id,
                    "category": category_name,
                    "constructs": group,
                }
            )
            pseudogene_counter += 1

        print(
            f"  Created {pseudogene_counter - 1} pseudo-genes for category '{category_name}'"
        )

    print(f"Total pseudo-genes created: {len(pseudogene_groups)}")
    print(f"Remaining individual constructs: {len(remaining_constructs)}")

    return pseudogene_groups, remaining_constructs


def write_pseudogene_data(pseudogene_group, output_dir):
    """Write construct data file for a pseudo-gene group."""
    pseudogene_id = pseudogene_group["pseudogene_id"]
    constructs = pseudogene_group["constructs"]

    print(f"  Writing pseudo-gene: {pseudogene_id} with {len(constructs)} constructs")

    # Create combined ID for the pseudo-gene
    combined_id = f"{pseudogene_id}__{pseudogene_id}"

    # Create metadata file for the pseudo-gene
    construct_data = pd.DataFrame(
        {
            "construct_id": [pseudogene_id],
            "gene": [pseudogene_id],
            "combined_id": [combined_id],
        }
    )

    # Save using combined_id for filename
    output_file = output_dir / f"{combined_id}__construct_data.tsv"
    construct_data.to_csv(output_file, sep="\t", index=False)


def write_construct_data(
    construct_id,
    construct_features_df,
    perturbation_col,
    perturbation_id_col,
    output_dir,
):
    """Write construct data file for a single construct."""
    # Get the gene for this construct
    construct_row = construct_features_df[
        construct_features_df[perturbation_id_col] == construct_id
    ]
    if len(construct_row) > 0:
        gene = construct_row[perturbation_col].iloc[0]
    else:
        gene = "unknown"

    # Create gene__construct combined ID
    combined_id = f"{gene}__{construct_id}"

    # Create metadata file for the construct
    construct_data = pd.DataFrame(
        {"construct_id": [construct_id], "gene": [gene], "combined_id": [combined_id]}
    )

    # Save using combined_id for filename
    output_file = output_dir / f"{combined_id}__construct_data.tsv"
    construct_data.to_csv(output_file, sep="\t", index=False)


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


def apply_multiple_hypothesis_correction(df, feature_cols):
    """Apply multiple hypothesis testing correction to p-values."""
    print("Applying multiple hypothesis testing correction...")

    # Create copies for corrected values
    df_corrected = df.copy()

    for feature in feature_cols:
        print(f"Processing feature: {feature}")

        # Get p-values for this feature
        pvals = df[feature].values

        # Handle missing/invalid p-values
        valid_mask = pd.notna(pvals) & (pvals > 0) & (pvals <= 1)

        if valid_mask.sum() == 0:
            print(f"  No valid p-values found for {feature}")
            df_corrected[f"{feature}_log10"] = np.nan
            df_corrected[f"{feature}_fdr"] = np.nan
            continue

        print(f"  Found {valid_mask.sum()} valid p-values out of {len(pvals)}")

        # Convert to -log10 (handle zeros by setting to minimum detectable p-value)
        log10_pvals = np.full(len(pvals), np.nan)

        # For valid p-values > 0
        nonzero_mask = valid_mask & (pvals > 0)
        if nonzero_mask.sum() > 0:
            log10_pvals[nonzero_mask] = -np.log10(pvals[nonzero_mask])

        # For p-values that are exactly 0 (perfect separation), set to high value
        zero_mask = valid_mask & (pvals == 0)
        if zero_mask.sum() > 0:
            print(
                f"  Found {zero_mask.sum()} p-values of exactly 0, setting to -log10(p) = 6"
            )
            log10_pvals[zero_mask] = 6.0  # Equivalent to p = 1e-6

        # Cap -log10 p-values at 4 (equivalent to p = 1e-4)
        log10_pvals = np.where(log10_pvals > 4, 4, log10_pvals)

        # Apply FDR correction (Benjamini-Hochberg)
        fdr_pvals = np.full(len(pvals), np.nan)
        if valid_mask.sum() > 1:  # Need at least 2 valid p-values for FDR
            try:
                # Extract valid p-values
                valid_pvals = pvals[valid_mask]

                # Apply FDR correction
                fdr_corrected = false_discovery_control(valid_pvals, method="bh")

                # Put corrected values back
                fdr_pvals[valid_mask] = fdr_corrected

                print(
                    f"  FDR correction applied. Significant at 0.05: {(fdr_corrected < 0.05).sum()}"
                )

            except Exception as e:
                print(f"  FDR correction failed for {feature}: {e}")
        else:
            print(f"  Insufficient valid p-values for FDR correction on {feature}")

        # Add corrected columns
        df_corrected[f"{feature}_log10"] = log10_pvals
        df_corrected[f"{feature}_fdr"] = fdr_pvals

    # Reorder columns by feature grouping
    print("Reordering columns by feature...")

    # Get metadata columns (everything that's not a feature or derived column)
    all_cols = df_corrected.columns.tolist()
    derived_cols = []
    for feature in feature_cols:
        derived_cols.extend([f"{feature}_log10", f"{feature}_fdr"])

    metadata_cols = [
        col for col in all_cols if col not in feature_cols and col not in derived_cols
    ]

    # Rename original feature columns to _pval
    rename_dict = {feature: f"{feature}_pval" for feature in feature_cols}
    df_corrected = df_corrected.rename(columns=rename_dict)

    # Build ordered column list: metadata + feature groups
    ordered_cols = metadata_cols.copy()

    for feature in feature_cols:
        ordered_cols.extend([f"{feature}_pval", f"{feature}_log10", f"{feature}_fdr"])

    # Return reordered dataframe
    return df_corrected[ordered_cols]
