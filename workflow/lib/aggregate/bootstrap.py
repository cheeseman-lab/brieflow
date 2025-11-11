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
from typing import Tuple, List, Dict, Any, Optional, Union
from scipy.stats import false_discovery_control


def get_construct_features(
    construct_id: str, construct_features_arr: np.ndarray
) -> np.ndarray:
    """Extract feature array for a specific construct.

    Args:
        construct_id (str): Identifier for the construct.
        construct_features_arr (np.ndarray): Array containing all construct features.

    Returns:
        np.ndarray: Feature array for the specified construct (excluding ID column).

    Raises:
        ValueError: If the construct is not found in the features array.
    """
    indices = np.where(construct_features_arr[:, 0] == construct_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Construct {construct_id} not found in features array")
    construct_arr = construct_features_arr[indices][0][1:].astype(float)
    return construct_arr


def create_pseudogene_groups(
    construct_features_df: pd.DataFrame,
    pseudogene_patterns: Dict[str, Dict[str, Union[str, int]]],
    perturbation_col: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Group constructs into pseudo-genes based on patterns.

    Args:
        construct_features_df (pd.DataFrame): DataFrame containing construct features.
        pseudogene_patterns (Dict[str, Dict[str, Union[str, int]]]): Dictionary mapping
            category names to pattern configurations. Each configuration should contain
            'pattern' (str) and 'constructs_per_pseudogene' (int) keys.
        perturbation_col (str): Name of the column containing perturbation identifiers.
        seed (int, optional): Random seed for reproducible grouping. Defaults to 42.

    Returns:
        Tuple[List[Dict[str, Any]], pd.DataFrame]: A tuple containing:
            - List of pseudo-gene group dictionaries with 'pseudogene_id', 'category',
              and 'constructs' keys
            - DataFrame of remaining individual constructs not grouped into pseudo-genes
    """
    if not pseudogene_patterns:
        return [], construct_features_df

    print(f"Pseudogene patterns provided: {pseudogene_patterns}")
    print(f"Using random seed: {seed}")

    # Set random seed for reproducible grouping
    random.seed(seed)

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

        # Sort constructs by ID for deterministic ordering before shuffle
        construct_list = matching_constructs.sort_values(perturbation_col).to_dict(
            "records"
        )

        # Shuffle with seeded random state
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


def write_construct_data(
    construct_id: str,
    construct_features_df: pd.DataFrame,
    perturbation_col: str,
    perturbation_id_col: str,
    output_dir: Path,
) -> None:
    """Write construct data file for a single construct.

    Args:
        construct_id (str): Identifier for the construct.
        construct_features_df (pd.DataFrame): DataFrame containing construct features.
        perturbation_col (str): Name of the column containing perturbation/gene names.
        perturbation_id_col (str): Name of the column containing perturbation IDs.
        output_dir (Path): Directory where the construct data file will be saved.
    """
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
        controls_arr (np.ndarray): Array of control data where first column contains
            identifiers and remaining columns contain features.
        sample_size (int): Size of sample to draw for the simulation.

    Returns:
        np.ndarray: Median values from the simulation across all features.
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
        null_medians_arr (np.ndarray): Array of null distribution medians from
            bootstrap simulations. Shape: (n_simulations, n_features).
        observed_medians (np.ndarray): Array of observed median values for each feature.
            Shape: (n_features,).

    Returns:
        np.ndarray: Array of two-tailed p-values for each feature.
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
        construct_id (str): Identifier for the construct to analyze.
        construct_features_arr (np.ndarray): Array containing all construct features
            where first column contains construct IDs and remaining columns contain features.
        controls_arr (np.ndarray): Array of control data where first column contains
            control IDs and remaining columns contain features.
        sample_size (int): Sample size for each bootstrap simulation.
        num_sims (int): Number of bootstrap simulations to run.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - null_medians_arr: Array of null distribution medians from all simulations.
              Shape: (num_sims, n_features)
            - p_vals: Array of p-values for each feature. Shape: (n_features,)

    Raises:
        ValueError: If bootstrap simulation fails due to constant values across simulations.
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
        file_paths (List[str]): List of file paths to null distribution numpy arrays.

    Returns:
        List[np.ndarray]: List of loaded numpy arrays containing null distributions.
    """
    return [np.load(path, allow_pickle=False) for path in file_paths]


def apply_multiple_hypothesis_correction(
    df: pd.DataFrame, feature_cols: List[str], min_p_value: Optional[float] = None
) -> pd.DataFrame:
    """Apply multiple hypothesis testing correction to p-values.

    Args:
        df (pd.DataFrame): DataFrame containing p-values to be corrected.
        feature_cols (List[str]): List of column names containing p-values to correct.
        min_p_value (Optional[float], optional): Minimum detectable p-value for FDR
            correction. If None, will be auto-detected from 'num_sims' column or
            default to 1e-5. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each feature:
            - {feature}_pval: Original p-values (including 0s)
            - {feature}_log10: -log10(p-values), with ceiling for p=0 cases
            - {feature}_fdr: FDR-corrected p-values using Benjamini-Hochberg method
    """
    print("Applying multiple hypothesis testing correction...")

    # Auto-detect minimum p-value for FDR correction (but keep original p=0 values)
    if min_p_value is None:
        # Look for num_sims column to determine minimum detectable p-value
        if "num_sims" in df.columns:
            min_p_value = 1.0 / df["num_sims"].max()
            max_log10 = np.log10(df["num_sims"].max())
            print(
                f"For FDR correction, treating p=0 as {min_p_value} (-log10 = {max_log10})"
            )
        else:
            min_p_value = 1e-5  # Conservative default (corresponds to -log10 = 5)
            max_log10 = 5.0
            print(
                f"Using default: treating p=0 as {min_p_value} for FDR, -log10 = {max_log10}"
            )

    df_corrected = df.copy()

    # First pass: collect all valid p-values for global FDR correction
    all_pvals = []
    pval_positions = []  # Track which feature and row each p-value came from

    for feature in feature_cols:
        pvals = df[feature].values
        valid_mask = pd.notna(pvals) & (pvals >= 0) & (pvals <= 1)

        for i, (pval, valid) in enumerate(zip(pvals, valid_mask)):
            if valid:
                # Handle p-values of exactly 0 (perfect separation in bootstrap)
                adjusted_pval = max(pval, min_p_value) if pval == 0 else pval
                all_pvals.append(adjusted_pval)
                pval_positions.append((feature, i))

    print(f"Found {len(all_pvals)} valid p-values across all features")

    # Apply global FDR correction to all p-values
    if len(all_pvals) > 1:
        try:
            fdr_corrected_all = false_discovery_control(all_pvals, method="bh")
            print(
                f"Global FDR correction applied. Significant at 0.05: {(fdr_corrected_all < 0.05).sum()}"
            )
        except Exception as e:
            print(f"Global FDR correction failed: {e}")
            fdr_corrected_all = all_pvals  # Fallback to uncorrected
    else:
        print("Insufficient p-values for FDR correction")
        fdr_corrected_all = all_pvals

    # Initialize output columns
    for feature in feature_cols:
        df_corrected[f"{feature}_log10"] = np.nan
        df_corrected[f"{feature}_fdr"] = np.nan

    # Second pass: populate corrected values
    fdr_idx = 0
    for feature in feature_cols:
        print(f"Processing feature: {feature}")
        pvals = df[feature].values
        valid_mask = pd.notna(pvals) & (pvals >= 0) & (pvals <= 1)

        if valid_mask.sum() == 0:
            print(f"  No valid p-values found for {feature}")
            continue

        print(f"  Found {valid_mask.sum()} valid p-values")

        # Process each p-value
        log10_pvals = np.full(len(pvals), np.nan)
        fdr_pvals = np.full(len(pvals), np.nan)

        for i, (pval, valid) in enumerate(zip(pvals, valid_mask)):
            if valid:
                # Calculate -log10(p): use ceiling for p=0, otherwise normal calculation
                if pval == 0:
                    # Set to log10(num_sims) if available, otherwise 5.0
                    if "num_sims" in df.columns:
                        log10_pvals[i] = np.log10(df.iloc[i]["num_sims"])
                    else:
                        log10_pvals[i] = 5.0
                else:
                    log10_pvals[i] = -np.log10(pval)

                # Get corresponding FDR-corrected p-value
                fdr_pvals[i] = fdr_corrected_all[fdr_idx]
                fdr_idx += 1

        # Store results (keep original p-values unchanged)
        df_corrected[f"{feature}_log10"] = log10_pvals
        df_corrected[f"{feature}_fdr"] = fdr_pvals

        # Report zeros found
        zero_count = (df[feature].values == 0).sum()
        if zero_count > 0:
            if "num_sims" in df.columns:
                max_log10 = np.log10(df["num_sims"].max())
                print(
                    f"  Found {zero_count} p-values of exactly 0 (kept as 0, -log10 = {max_log10:.1f})"
                )
            else:
                print(
                    f"  Found {zero_count} p-values of exactly 0 (kept as 0, -log10 = 5.0)"
                )

    # Reorder columns: metadata + feature groups
    print("Reordering columns by feature...")

    # Identify metadata columns
    all_cols = df_corrected.columns.tolist()
    derived_cols = []
    for feature in feature_cols:
        derived_cols.extend([f"{feature}_log10", f"{feature}_fdr"])

    metadata_cols = [
        col for col in all_cols if col not in feature_cols and col not in derived_cols
    ]

    # Rename original columns to _pval
    rename_dict = {feature: f"{feature}_pval" for feature in feature_cols}
    df_corrected = df_corrected.rename(columns=rename_dict)

    # Build ordered column list
    ordered_cols = metadata_cols.copy()
    for feature in feature_cols:
        ordered_cols.extend(
            [
                f"{feature}_pval",  # Original p-value (including 0s)
                f"{feature}_log10",  # -log10(p-value), with ceiling for p=0
                f"{feature}_fdr",  # FDR-corrected p-value
            ]
        )

    return df_corrected[ordered_cols]
