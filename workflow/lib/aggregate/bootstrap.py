"""Bootstrap statistical testing for perturbation effects.

This module provides functionality for performing bootstrap statistical tests
to validate perturbation effects detected in the aggregate pipeline. It includes
functions for data preparation, feature selection, bootstrap simulation, and
p-value calculation at both construct and gene levels.
"""

import zlib

import numpy as np
import pandas as pd
import random
import numba
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
from scipy.stats import false_discovery_control

from lib.aggregate.cell_data_utils import CONTROL_SCOPES, GROUP_KEY_SEP, control_mask

# Simulations per block; fixed so a gene's p-values do not depend on its size
BOOTSTRAP_BLOCK_SIMS = 1000


def select_control_pool(
    controls_df: pd.DataFrame,
    construct_id: str,
    control_scope: str = "pooled",
    reference_group: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
    perturbation_id: Optional[str] = None,
) -> pd.DataFrame:
    """Restrict the bootstrap null pool to the controls a scope names.

    "reference_group" pins the null to one fixed group whatever the construct's own
    group, so a library whose group is a treatment acting on the perturbation is
    tested against the untreated control state rather than against controls that saw
    the same treatment.

    "within_perturbation" goes one step further and makes every perturbation its own
    control: the null for `X=treated` is drawn from `X=reference_group`, the same
    perturbation in the reference condition. The pool then holds every cell in the
    reference group rather than the control_key perturbations (see
    bootstrap_control_mask), and the control is the reference condition rather than
    any perturbation.

    Args:
        controls_df (pd.DataFrame): Control cells, perturbation key in the first column.
        construct_id (str): Construct being tested, composite when group_cols is set.
        control_scope (str, optional): "pooled" keeps every control, "within_group"
            keeps controls sharing the construct's own group, "reference_group" keeps
            controls in `reference_group`, "within_perturbation" keeps cells of the
            construct's own perturbation in `reference_group`. Defaults to "pooled".
        reference_group (str, optional): Group key the "reference_group" and
            "within_perturbation" scopes pin the pool to; several group_cols join
            their values with GROUP_KEY_SEP. Defaults to None.
        group_cols (list, optional): Columns folded into the composite key.
            Defaults to None.
        perturbation_id (str, optional): The construct's perturbation-level key, as
            written in the first column of controls_df. Required by
            "within_perturbation", which matches on it since construct ids carry a
            guide or barcode instead. Defaults to None.

    Returns:
        pd.DataFrame: Controls the scope selects.

    Raises:
        ValueError: If the scope is unknown, if a reference scope is requested without
            group_cols or without a reference_group, if "within_perturbation" is
            requested without a perturbation_id, or if the selected pool is empty.
    """
    if control_scope not in CONTROL_SCOPES:
        raise ValueError(f"Unknown bootstrap_control_scope: {control_scope}")

    if control_scope == "pooled":
        return controls_df

    if control_scope == "within_group":
        if GROUP_KEY_SEP not in str(construct_id):
            return controls_df
        group_key = str(construct_id).split(GROUP_KEY_SEP, 1)[1]
    else:
        if not group_cols:
            raise ValueError(
                f"bootstrap_control_scope '{control_scope}' needs aggregate "
                "group_cols; without them controls carry no group to pin the null to"
            )
        if reference_group is None:
            raise ValueError(
                f"bootstrap_control_scope '{control_scope}' requires "
                "bootstrap_reference_group to be set"
            )
        group_key = str(reference_group)

    control_keys = controls_df.iloc[:, 0].astype(str).str.split(GROUP_KEY_SEP, n=1)
    control_groups = control_keys.str[1]
    group_mask = control_groups == group_key

    if control_scope == "within_perturbation":
        if perturbation_id is None:
            raise ValueError(
                "bootstrap_control_scope 'within_perturbation' needs the construct's "
                f"perturbation_id (construct {construct_id})"
            )
        own_perturbation = str(perturbation_id).split(GROUP_KEY_SEP, 1)[0]
        group_mask &= control_keys.str[0] == own_perturbation
        label = f"perturbation '{own_perturbation}' in group '{group_key}'"
    else:
        label = f"group '{group_key}'"

    n_selected = int(group_mask.sum())
    print(f"Restricting controls to {label}: {n_selected} of {len(controls_df)} rows")
    control_pool = controls_df[group_mask]
    if len(control_pool) == 0:
        if control_scope == "reference_group":
            raise ValueError(
                f"bootstrap_reference_group '{group_key}' is absent from the control pool; "
                f"control groups present: {sorted(control_groups.dropna().unique())}"
            )
        raise ValueError(
            f"No control cells found for {label} (construct {construct_id})"
        )

    return control_pool


def bootstrap_control_mask(
    cells: pd.DataFrame,
    perturbation_col: str,
    control_key: Union[str, List[str]],
    control_scope: str = "pooled",
    reference_group: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
) -> pd.Series:
    """Flag the cells written to the bootstrap control pool.

    Every scope but "within_perturbation" pools the control_key perturbations and
    select_control_pool narrows them per construct. "within_perturbation" pools every
    cell in the reference group instead, since each perturbation's null is its own
    arm there.

    Args:
        cells (pd.DataFrame): Single-cell data carrying perturbation_col and the
            literal group_cols.
        perturbation_col (str): Perturbation name column.
        control_key (str | list): Control identifier, or a list of exact names.
        control_scope (str, optional): Bootstrap control scope. Defaults to "pooled".
        reference_group (str, optional): Group key of the reference condition; several
            group_cols join their values with GROUP_KEY_SEP. Defaults to None.
        group_cols (list, optional): Columns folded into the composite key.
            Defaults to None.

    Returns:
        pd.Series: Boolean mask of the cells in the control pool.

    Raises:
        ValueError: If "within_perturbation" is requested without group_cols or
            without a reference_group, or if no cell is in the reference group.
    """
    if control_scope != "within_perturbation":
        return control_mask(cells[perturbation_col], control_key)

    if not group_cols or reference_group is None:
        raise ValueError(
            "bootstrap_control_scope 'within_perturbation' needs aggregate group_cols "
            "and bootstrap_reference_group"
        )
    groups = cells[list(group_cols)].astype(str).agg(GROUP_KEY_SEP.join, axis=1)
    in_reference = groups == str(reference_group)
    if not in_reference.any():
        raise ValueError(
            f"bootstrap_reference_group '{reference_group}' matches no cells; "
            f"groups present: {sorted(groups.unique())}"
        )
    print(
        f"Control pool is the reference group '{reference_group}': "
        f"{int(in_reference.sum())} cells"
    )

    return in_reference


def within_perturbation_construct_mask(
    construct_table: pd.DataFrame,
    perturbation_col: str,
    control_perturbations: pd.Series,
    reference_group: str,
) -> pd.Series:
    """Flag the constructs a "within_perturbation" bootstrap can test.

    A construct is tested only when its perturbation has an arm in the reference group,
    since that arm is its null, and only when it is not in the reference group itself:
    a reference arm drawn against its own cells is a circular test that would also
    enlarge the multiple-testing correction applied to every other construct.

    Args:
        construct_table (pd.DataFrame): Construct-level table, composite keys in
            perturbation_col.
        perturbation_col (str): Perturbation name column.
        control_perturbations (pd.Series): Composite perturbation keys of the cells in
            the control pool, as bootstrap_control_mask selects them.
        reference_group (str): Group key of the reference condition.

    Returns:
        pd.Series: Boolean mask aligned with construct_table.
    """
    construct_keys = (
        construct_table[perturbation_col].astype(str).str.split(GROUP_KEY_SEP, n=1)
    )
    construct_perturbations = construct_keys.str[0]
    referenced = set(
        control_perturbations.astype(str).str.split(GROUP_KEY_SEP, n=1).str[0]
    )
    has_reference = construct_perturbations.isin(referenced)
    is_reference = construct_keys.str[1] == str(reference_group)

    unreferenced = sorted(construct_perturbations[~has_reference].unique())
    if unreferenced:
        print(
            f"{len(unreferenced)} perturbations have no '{reference_group}' arm "
            f"and are left out of the bootstrap: {unreferenced}"
        )
    print(
        f"{int(is_reference.sum())} constructs in '{reference_group}' are the null "
        "and are not tested"
    )

    return has_reference & ~is_reference


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
    exceed_counts = (null_medians_arr > observed_medians).sum(axis=0)
    return pvals_from_exceed_counts(exceed_counts, null_medians_arr.shape[0])


def run_gene_bootstrap(
    construct_observed: np.ndarray,
    construct_controls: List[np.ndarray],
    sample_sizes: List[int],
    gene_observed: np.ndarray,
    num_sims: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap every construct of one gene and the gene itself, keeping only exceed counts.

    Each construct's null draws `sample_size` control cells with replacement from the
    pool of one control perturbation (picked with probability proportional to its size)
    and takes the per-feature median. The gene's null is, simulation by simulation, the
    median across its constructs' nulls. Simulations run in blocks so memory stays
    bounded, and only the count of null medians above the observed value is kept, so no
    null distribution is written or held beyond one block.

    Args:
        construct_observed (np.ndarray): (n_constructs, n_features) observed construct medians.
        construct_controls (List[np.ndarray]): Per construct, its control pool: first
            column the control perturbation id, remaining columns the features.
        sample_sizes (List[int]): Per construct, cells sampled per simulation.
        gene_observed (np.ndarray): (n_features,) observed gene medians.
        num_sims (int): Number of bootstrap simulations.
        seed (int): Seed for the simulation random state, see `bootstrap_seed`.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two-tailed p-values, (n_constructs, n_features)
            for the constructs and (n_features,) for the gene.

    Raises:
        ValueError: If a block of simulations returns constant values.
    """
    construct_observed = np.asarray(construct_observed, dtype=np.float64)
    gene_observed = np.asarray(gene_observed, dtype=np.float64)
    n_constructs, n_features = construct_observed.shape
    pools = [_control_pool_arrays(controls) for controls in construct_controls]
    block_size = min(num_sims, BOOTSTRAP_BLOCK_SIMS)
    nulls = np.empty((n_constructs, block_size, n_features), dtype=np.float64)
    construct_exceed = np.zeros((n_constructs, n_features), dtype=np.int64)
    gene_exceed = np.zeros(n_features, dtype=np.int64)

    _seed_numba(seed)
    done = 0
    while done < num_sims:
        n = min(block_size, num_sims - done)
        for c, (features_t, pool_lookup, pool_starts, pool_ends) in enumerate(pools):
            block = nulls[c, :n]
            _bootstrap_block_numba(
                features_t, pool_lookup, pool_starts, pool_ends, sample_sizes[c], block
            )
            _check_simulations(block)
            construct_exceed[c] += (block > construct_observed[c]).sum(axis=0)
        gene_nulls = np.median(nulls[:, :n], axis=0)
        gene_exceed += (gene_nulls > gene_observed).sum(axis=0)
        done += n

    return (
        pvals_from_exceed_counts(construct_exceed, num_sims),
        pvals_from_exceed_counts(gene_exceed, num_sims),
    )


def read_id_keyed_tsv(fp: Union[str, Path]) -> pd.DataFrame:
    """Read a bootstrap input table indexed by its first column, read as strings.

    Construct ids are compared as strings with the ids in the checkpoint filenames, so
    the id column is never type-inferred (a numeric-looking id keeps its leading zeros).

    Args:
        fp (Union[str, Path]): Tab-separated table with the id in its first column.

    Returns:
        pd.DataFrame: The table indexed by the string id column.
    """
    id_col = pd.read_csv(fp, sep="\t", nrows=0).columns[0]
    return pd.read_csv(fp, sep="\t", dtype={id_col: str}).set_index(id_col)


def pvals_from_exceed_counts(exceed_counts: np.ndarray, num_sims: int) -> np.ndarray:
    """Convert counts of null medians above the observed value into two-tailed p-values.

    Args:
        exceed_counts (np.ndarray): Per feature, how many simulations exceeded the
            observed median.
        num_sims (int): Number of simulations the counts were taken over.

    Returns:
        np.ndarray: Two-tailed p-values, same shape as `exceed_counts`.
    """
    pvals_one_tail = np.asarray(exceed_counts, dtype=np.float64) / num_sims
    return np.where(pvals_one_tail > 0.5, 1 - pvals_one_tail, pvals_one_tail) * 2


def bootstrap_seed(gene_id: str) -> int:
    """Stable seed for one gene's simulations, so a rerun reproduces its p-values.

    Args:
        gene_id (str): Gene (or composite gene key) being tested.

    Returns:
        int: 32-bit seed derived from the gene id.
    """
    return zlib.crc32(str(gene_id).encode("utf-8"))


def _control_pool_arrays(
    controls_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lay out a control pool for the simulation kernel.

    Rows are sorted by control id so each control perturbation occupies a contiguous
    slice, and features are stored feature-major for cache-friendly sampling.

    Args:
        controls_arr (np.ndarray): Control id in the first column, features after.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (n_features, n_rows)
            features, row-to-pool lookup, pool start rows, pool end rows (exclusive).
    """
    control_ids = controls_arr[:, 0]
    controls_features = controls_arr[:, 1:].astype(np.float64)
    sort_order = np.argsort(control_ids, kind="stable")
    features_t = np.ascontiguousarray(controls_features[sort_order].T)
    _, counts = np.unique(control_ids[sort_order], return_counts=True)
    pool_ends = np.cumsum(counts).astype(np.int64)
    pool_starts = np.empty_like(pool_ends)
    pool_starts[0] = 0
    pool_starts[1:] = pool_ends[:-1]
    pool_lookup = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return features_t, pool_lookup, pool_starts, pool_ends


def _check_simulations(null_medians: np.ndarray) -> None:
    """Raise if a block of simulations returned constant values.

    Args:
        null_medians (np.ndarray): (n_sims, n_features) null medians of one block.

    Raises:
        ValueError: If the simulations are degenerate.
    """
    if null_medians.shape[1] == 1:
        if null_medians.shape[0] > 1 and len(np.unique(null_medians)) == 1:
            raise ValueError("All bootstrap simulations returned identical values")
    elif np.any(np.max(null_medians, axis=1) == np.min(null_medians, axis=1)):
        raise ValueError(
            "Bootstrap simulation failed - some simulations returned constant values"
        )


@numba.njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


@numba.njit(cache=True)
def _bootstrap_block_numba(
    features_t, pool_lookup, pool_starts, pool_ends, sample_size, out
):
    """Fill `out` (n_sims, n_features) with null medians, one simulation per row.

    Each simulation picks a random control row to choose its pool, samples
    `sample_size` rows with replacement from that pool, and takes the median of
    each feature, as np.median does (NaN if any sampled value is NaN).
    """
    n_features, n_rows = features_t.shape
    rows = np.empty(sample_size, dtype=np.int64)
    buf = np.empty(sample_size, dtype=np.float64)
    for i in range(out.shape[0]):
        seed_row = np.random.randint(0, n_rows)
        pool_idx = pool_lookup[seed_row]
        start = pool_starts[pool_idx]
        pool_size = pool_ends[pool_idx] - start
        for k in range(sample_size):
            rows[k] = start + np.random.randint(0, pool_size)
        for j in range(n_features):
            has_nan = False
            for k in range(sample_size):
                value = features_t[j, rows[k]]
                buf[k] = value
                if np.isnan(value):
                    has_nan = True
            out[i, j] = np.nan if has_nan else _median_inplace(buf)


@numba.njit(cache=True)
def _median_inplace(buf):
    """Median of `buf` by in-place selection; reorders `buf`."""
    n = buf.shape[0]
    k = n // 2
    lo = 0
    hi = n - 1
    while lo < hi:
        pivot = buf[k]
        i = lo
        j = hi
        while i <= j:
            while buf[i] < pivot:
                i += 1
            while pivot < buf[j]:
                j -= 1
            if i <= j:
                tmp = buf[i]
                buf[i] = buf[j]
                buf[j] = tmp
                i += 1
                j -= 1
        if j < k:
            lo = i
        if k < i:
            hi = j
    upper = buf[k]
    if n % 2 == 1:
        return upper
    lower = buf[0]
    for t in range(1, k):
        if buf[t] > lower:
            lower = buf[t]
    return (lower + upper) / 2


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
            - {feature}_neg_log10_pval: -log10(p-values), with ceiling for p=0 cases
            - {feature}_fdr: FDR-corrected p-values using Benjamini-Hochberg method
            - {feature}_neg_log10_fdr: -log10(FDR-corrected p-values)
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
        df_corrected[f"{feature}_neg_log10_pval"] = np.nan
        df_corrected[f"{feature}_fdr"] = np.nan
        df_corrected[f"{feature}_neg_log10_fdr"] = np.nan

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
        neg_log10_pvals = np.full(len(pvals), np.nan)
        fdr_pvals = np.full(len(pvals), np.nan)

        for i, (pval, valid) in enumerate(zip(pvals, valid_mask)):
            if valid:
                # Calculate -log10(p): use ceiling for p=0, otherwise normal calculation
                if pval == 0:
                    if "num_sims" in df.columns:
                        neg_log10_pvals[i] = np.log10(df.iloc[i]["num_sims"])
                    else:
                        neg_log10_pvals[i] = 5.0
                else:
                    neg_log10_pvals[i] = -np.log10(pval)

                # Get corresponding FDR-corrected p-value
                fdr_pvals[i] = fdr_corrected_all[fdr_idx]
                fdr_idx += 1

        # Compute -log10(FDR)
        neg_log10_fdr = np.full(len(fdr_pvals), np.nan)
        for i, fdr_val in enumerate(fdr_pvals):
            if not np.isnan(fdr_val) and fdr_val > 0:
                neg_log10_fdr[i] = -np.log10(fdr_val)
            elif not np.isnan(fdr_val) and fdr_val == 0:
                neg_log10_fdr[i] = max_log10

        # Store results (keep original p-values unchanged)
        df_corrected[f"{feature}_neg_log10_pval"] = neg_log10_pvals
        df_corrected[f"{feature}_fdr"] = fdr_pvals
        df_corrected[f"{feature}_neg_log10_fdr"] = neg_log10_fdr

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
        derived_cols.extend(
            [
                f"{feature}_neg_log10_pval",
                f"{feature}_fdr",
                f"{feature}_neg_log10_fdr",
            ]
        )

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
                f"{feature}_pval",
                f"{feature}_neg_log10_pval",
                f"{feature}_fdr",
                f"{feature}_neg_log10_fdr",
            ]
        )

    return df_corrected[ordered_cols]
