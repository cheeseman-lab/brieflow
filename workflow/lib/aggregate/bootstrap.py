"""Bootstrap statistical testing for perturbation effects.

This module provides functionality for performing bootstrap statistical tests
to validate perturbation effects detected in the aggregate pipeline. It includes
functions for data preparation, feature selection, bootstrap simulation, and
p-value calculation at both construct and gene levels.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any


def select_bootstrap_features(feature_cols: List[str]) -> List[str]:
    """Select features for bootstrap analysis based on feature types.
    
    Selects features containing 'area', 'median', 'mean' (excluding 'frac' and 'edge'),
    or 'integrated' in their names.
    
    Args:
        feature_cols: List of feature column names.
        
    Returns:
        List of selected feature column names.
    """
    selected_features = []
    
    for col in feature_cols:
        if 'area' in col:
            selected_features.append(col)
        elif 'median' in col:
            selected_features.append(col)
        elif 'mean' in col and ('frac' not in col and 'edge' not in col):
            selected_features.append(col)
        elif 'integrated' in col:
            selected_features.append(col)
    
    return selected_features


def prep_bootstrap_data(
    cell_data: pd.DataFrame,
    metadata_cols: List[str],
    perturbation_col: str,
    control_key: str,
    sample_size_col: str,
    exclusion_string: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """Prepare data arrays for bootstrap analysis.
    
    Args:
        cell_data: DataFrame containing cell-level data with metadata and features.
        metadata_cols: List of metadata column names.
        perturbation_col: Column name containing perturbation identifiers.
        control_key: String identifying control perturbations.
        sample_size_col: Column name containing sample sizes.
        exclusion_string: Optional string to exclude from perturbations.
        
    Returns:
        Tuple containing:
        - controls_arr: Array of control cell data
        - construct_features_arr: Array of construct feature data
        - sample_sizes_df: DataFrame with sample sizes per construct
        - selected_features: List of selected feature names
    """
    # Split metadata and features
    metadata = cell_data[metadata_cols].copy()
    all_features = cell_data.drop(columns=metadata_cols)
    
    # Select bootstrap-relevant features
    selected_features = select_bootstrap_features(all_features.columns.tolist())
    features = all_features[selected_features].copy()
    
    # Get controls array
    controls_mask = metadata[perturbation_col].str.contains(control_key, na=False)
    controls_df = pd.concat([metadata[controls_mask][[perturbation_col]], 
                            features[controls_mask]], axis=1)
    controls_arr = controls_df.values
    
    # Get constructs array (exclude controls and exclusion string if provided)
    constructs_mask = ~metadata[perturbation_col].str.contains(control_key, na=False)
    if exclusion_string is not None:
        constructs_mask = constructs_mask & ~metadata[perturbation_col].str.contains(exclusion_string, na=False)
    
    constructs_df = pd.concat([metadata[constructs_mask][[perturbation_col]], 
                              features[constructs_mask]], axis=1)
    construct_features_arr = constructs_df.values
    
    # Get sample sizes
    sample_sizes_df = metadata[constructs_mask][[perturbation_col, sample_size_col]].copy()
    
    return controls_arr, construct_features_arr, sample_sizes_df, selected_features


def get_construct_features(
    construct_id: str,
    construct_features_arr: np.ndarray
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


def get_sample_size(
    sample_sizes_df: pd.DataFrame,
    construct_id: str,
    sample_size_col: str
) -> int:
    """Get sample size for a specific construct.
    
    Args:
        sample_sizes_df: DataFrame containing sample sizes.
        construct_id: Identifier for the construct.
        sample_size_col: Column name containing sample sizes.
        
    Returns:
        Sample size for the construct.
    """
    mask = sample_sizes_df.iloc[:, 0] == construct_id  # First column is perturbation ID
    if not mask.any():
        raise ValueError(f"Construct {construct_id} not found in sample sizes")
    sample_size = sample_sizes_df.loc[mask, sample_size_col].iloc[0]
    return int(sample_size)


def sample_control_construct(controls_arr: np.ndarray) -> np.ndarray:
    """Randomly sample a control construct from controls array.
    
    Args:
        controls_arr: Array of control data.
        
    Returns:
        Array of data for the sampled control construct.
    """
    control_id = controls_arr[np.random.choice(controls_arr.shape[0], 1, replace=False)][0][0]
    indices = np.where(controls_arr[:, 0] == control_id)[0]
    control_construct_arr = controls_arr[indices]
    return control_construct_arr


def sample_null_distribution(
    control_construct_arr: np.ndarray,
    sample_size: int
) -> np.ndarray:
    """Sample from null distribution using control construct.
    
    Args:
        control_construct_arr: Array of control construct data.
        sample_size: Number of samples to draw.
        
    Returns:
        Array of sampled values (excluding ID column).
    """
    sample_arr = control_construct_arr[
        np.random.choice(control_construct_arr.shape[0], sample_size, replace=True)
    ]
    return sample_arr[:, 1:].astype(float)


def compute_sample_median(sample_arr: np.ndarray) -> np.ndarray:
    """Compute median across samples.
    
    Args:
        sample_arr: Array of sampled values.
        
    Returns:
        Median values across samples.
    """
    return np.median(sample_arr, axis=0)


def run_single_bootstrap_simulation(
    controls_arr: np.ndarray,
    sample_size: int
) -> np.ndarray:
    """Run a single bootstrap simulation.
    
    Args:
        controls_arr: Array of control data.
        sample_size: Size of sample to draw.
        
    Returns:
        Median values from the simulation.
    """
    control_construct_arr = sample_control_construct(controls_arr)
    sample_arr = sample_null_distribution(control_construct_arr, sample_size)
    median_arr = compute_sample_median(sample_arr)
    return median_arr


def calculate_pvals(
    null_medians_arr: np.ndarray,
    observed_medians: np.ndarray
) -> np.ndarray:
    """Calculate two-tailed p-values from bootstrap null distribution.
    
    Args:
        null_medians_arr: Array of null distribution medians from simulations.
        observed_medians: Array of observed median values.
        
    Returns:
        Array of two-tailed p-values.
    """
    # One-tailed p-values
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
    num_sims: int
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
    
    # Verify simulations ran properly (no all-zero rows)
    all_same = np.max(null_medians_arr, axis=1) == np.min(null_medians_arr, axis=1)
    if len(null_medians_arr[all_same]) > 0:
        raise ValueError("Bootstrap simulation failed - some simulations returned constant values")
    
    # Calculate p-values
    p_vals = calculate_pvals(null_medians_arr, observed_medians)
    
    return null_medians_arr, p_vals


def format_construct_results(
    construct_id: str,
    p_vals: np.ndarray,
    feature_names: List[str],
    sample_size: int,
    num_sims: int
) -> pd.DataFrame:
    """Format bootstrap results for a construct into DataFrame.
    
    Args:
        construct_id: Identifier for the construct.
        p_vals: Array of p-values.
        feature_names: List of feature names.
        sample_size: Sample size used.
        num_sims: Number of simulations run.
        
    Returns:
        DataFrame with formatted results.
    """
    pval_df = pd.DataFrame(columns=['construct', 'sample_size', 'num_sims'] + feature_names)
    pval_df.loc[0] = [construct_id, sample_size, num_sims] + list(p_vals)
    return pval_df


def load_construct_null_arrays(file_paths: List[str]) -> List[np.ndarray]:
    """Load multiple construct null distribution arrays.
    
    Args:
        file_paths: List of paths to null distribution files.
        
    Returns:
        List of loaded numpy arrays.
    """
    return [np.load(path, allow_pickle=False) for path in file_paths]


def aggregate_gene_results(
    gene_id: str,
    construct_null_arrays: List[np.ndarray],
    gene_features_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Aggregate construct-level bootstrap results to gene level.
    
    Args:
        gene_id: Identifier for the gene.
        construct_null_arrays: List of null distribution arrays from constructs.
        gene_features_arr: Array containing gene-level features.
        
    Returns:
        Tuple containing:
        - median_null_medians: Median of construct null distributions
        - p_vals: Gene-level p-values
        - num_constructs: Number of constructs aggregated
    """
    print(f"Processing gene: {gene_id}")
    
    # Get observed gene medians
    gene_medians = get_construct_features(gene_id, gene_features_arr)
    
    print(f"Construct null array shapes: {[arr.shape for arr in construct_null_arrays]}")
    print(f"Gene medians shape: {gene_medians.shape}")
    
    # Stack and take median across constructs
    stacked_medians_arr = np.stack(construct_null_arrays)
    median_null_medians = np.median(stacked_medians_arr, axis=0)
    
    # Calculate p-values
    p_vals = calculate_pvals(median_null_medians, gene_medians)
    
    return median_null_medians, p_vals, len(construct_null_arrays)


def format_gene_results(
    gene_id: str,
    p_vals: np.ndarray,
    feature_names: List[str],
    num_constructs: int,
    num_sims: int
) -> pd.DataFrame:
    """Format bootstrap results for a gene into DataFrame.
    
    Args:
        gene_id: Identifier for the gene.
        p_vals: Array of p-values.
        feature_names: List of feature names.
        num_constructs: Number of constructs aggregated.
        num_sims: Number of simulations per construct.
        
    Returns:
        DataFrame with formatted results.
    """
    pval_df = pd.DataFrame(columns=['gene', 'num_constructs', 'num_sims'] + feature_names)
    pval_df.loc[0] = [gene_id, num_constructs, num_sims] + list(p_vals)
    return pval_df


def parse_gene_construct_mapping(construct_ids: List[str]) -> Dict[str, List[str]]:
    """Parse construct IDs to create gene-to-constructs mapping.
    
    Assumes construct IDs are in format 'gene.construct'.
    
    Args:
        construct_ids: List of construct identifiers.
        
    Returns:
        Dictionary mapping gene IDs to lists of construct IDs.
    """
    gene_dict = {}
    
    for construct_id in construct_ids:
        if '.' in construct_id:
            gene, construct = construct_id.split('.', 1)
            if gene not in gene_dict:
                gene_dict[gene] = []
            gene_dict[gene].append(construct)
        else:
            # If no dot separator, treat as gene name
            if construct_id not in gene_dict:
                gene_dict[construct_id] = [construct_id]
    
    return gene_dict
