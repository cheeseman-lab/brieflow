"""Module for automating parameter search in SBS.

This module provides functions to perform an automated parameter search
for SBS (sequencing by synthesis) processing. It tests different parameter
combinations and evaluates results using configurable metric functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

warnings.filterwarnings("ignore")


def automated_parameter_search(
    aligned_images,
    mask,
    barcodes,
    df_barcode_library,
    wildcards,
    bases,
    extra_channel_indices,
    param_grid,
    metric_fn=None,
    fixed_params=None,
    barcode_type="simple",
    verbose=False,
):
    """Perform automated parameter search for SBS processing.

    This function systematically tests combinations of parameters specified
    in param_grid and evaluates each combination using a metric function.

    Parameters
    ----------
    aligned_images : np.ndarray
        Array of aligned SBS images (cycles x channels x height x width).
    mask : np.ndarray
        Labeled segmentation mask where each cell/nucleus has a unique integer label.
        Can be either nuclei or cells mask depending on where spots are located.
    barcodes : list or np.ndarray
        List of barcode sequences to map reads against.
    df_barcode_library : pd.DataFrame
        DataFrame containing barcode library information for mapping.
    wildcards : dict
        Dictionary of wildcards for file naming (e.g., {'well': 'A1', 'tile': 1}).
    bases : list
        List of base labels (e.g., ['G', 'T', 'A', 'C']).
    extra_channel_indices : list or np.ndarray
        Indices of channels to skip during filtering (e.g., DAPI channel).
    param_grid : dict
        Dictionary mapping parameter names to lists of values to test.
        Example: {'peak_width': [5, 10, 15], 'threshold_reads': [50, 100, 150]}
    metric_fn : callable, optional
        Function that takes df_cells and returns a float score to optimize.
        If None, uses metric_one_barcode_fraction as default.
        Note: Built-in metrics (specificity, one_barcode_fraction, any_barcode_fraction)
        should be **maximized** (higher is better). Custom metrics should be designed
        similarly, or use get_best_parameters(maximize=False) if lower is better.
    fixed_params : dict, optional
        Dictionary of parameters that remain constant across all searches.
        Example: {'max_filter_width': 3, 'call_reads_method': 'percentile'}
        For multi-barcode mode, should include: map_start, map_end, recomb_start,
        recomb_end, map_col, recomb_col, recomb_filter_col, recomb_q_thresh
    barcode_type : str, default='simple'
        Type of barcode protocol: 'simple' or 'multi'
    verbose : bool, default=False
        If True, print progress and debug information.

    Returns:
    -------
    results_df : pd.DataFrame
        DataFrame with one row per parameter combination, including:
        - All parameter values
        - metric_score: the score from metric_fn
        - metric_name: name of the metric function used
        - status: 'success' or error description
        - total_cells: number of cells in mask
    df_cells_combined : pd.DataFrame or None
        Combined DataFrame of per-cell calls for all successful parameter sets,
        with parameter values annotated in each row.
    """
    # Set defaults
    if metric_fn is None:
        metric_fn = metric_one_barcode_fraction
    if fixed_params is None:
        fixed_params = {}

    # Get metric name from function
    metric_name = metric_fn.__name__
    # Remove 'metric_' prefix if present for cleaner display
    if metric_name.startswith("metric_"):
        metric_name = metric_name[7:]  # Remove 'metric_' prefix

    # Import required functions
    try:
        from lib.shared.log_filter import log_filter
        from lib.sbs.max_filter import max_filter
        from lib.sbs.compute_standard_deviation import compute_standard_deviation
        from lib.sbs.find_peaks import find_peaks
        from lib.sbs.extract_bases import extract_bases
        from lib.sbs.call_reads import call_reads
        from lib.sbs.call_cells import call_cells, prep_multi_reads
    except ImportError as e:
        print(f"Error importing required functions: {e}")
        return pd.DataFrame(), None

    results = []
    all_cells = []

    # Pre-process images (done once for all parameter combinations)
    if verbose:
        print("Pre-processing images...")

    try:
        max_filter_width = fixed_params.get("max_filter_width", 3)
        loged = log_filter(aligned_images, skip_index=extra_channel_indices)
        maxed = max_filter(
            loged, width=max_filter_width, remove_index=extra_channel_indices
        )
    except Exception as e:
        if verbose:
            print(f"Pre-processing failed: {e}")
        return pd.DataFrame(), None

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total_combinations = len(combinations)

    if verbose:
        print(f"Testing {total_combinations} parameter combinations...")
        print(f"Parameters: {param_names}")
        print(f"Metric: {metric_name}")

    # Test each combination
    for combo_idx, combo_values in enumerate(combinations, 1):
        # Create parameter dictionary for this combination
        current_params = dict(zip(param_names, combo_values))

        if verbose and combo_idx % 5 == 0:
            print(f"Progress: {combo_idx}/{total_combinations}")

        try:
            # Extract parameters for this iteration
            peak_width = current_params.get("peak_width", 5)
            threshold_reads = current_params.get("threshold_reads", 100)
            q_min = current_params.get("q_min", fixed_params.get("q_min", 0))
            call_reads_method = current_params.get(
                "call_reads_method", fixed_params.get("call_reads_method", "percentile")
            )
            error_correct = current_params.get(
                "error_correct", fixed_params.get("error_correct", True)
            )
            sort_calls = current_params.get(
                "sort_calls", fixed_params.get("sort_calls", "count")
            )

            # Compute standard deviation and find peaks
            standard_deviation = compute_standard_deviation(
                loged, remove_index=extra_channel_indices
            )
            peaks = find_peaks(standard_deviation, width=peak_width)

            if peaks is None or (hasattr(peaks, "__len__") and len(peaks) == 0):
                results.append(
                    {
                        **current_params,
                        "metric_score": 0.0,
                        "metric_name": metric_name,
                        "total_cells": len(np.unique(mask)) - 1,
                        "status": "no_peaks",
                    }
                )
                continue

            # Extract bases
            df_bases = extract_bases(
                peaks,
                maxed,
                mask,
                threshold_reads,
                wildcards=wildcards,
                bases=bases,
            )

            if df_bases is None or len(df_bases) == 0:
                results.append(
                    {
                        **current_params,
                        "metric_score": 0.0,
                        "metric_name": metric_name,
                        "total_cells": len(np.unique(mask)) - 1,
                        "status": "no_bases",
                    }
                )
                continue

            # Call reads
            df_reads = call_reads(df_bases, peaks_data=peaks, method=call_reads_method)

            if df_reads is None or len(df_reads) == 0:
                results.append(
                    {
                        **current_params,
                        "metric_score": 0.0,
                        "metric_name": metric_name,
                        "total_cells": len(np.unique(mask)) - 1,
                        "status": "no_reads",
                    }
                )
                continue

            # Call cells using the barcode library
            if barcode_type == "multi":
                # Multi-barcode mode: prep reads first, then call cells
                df_reads_prepped = prep_multi_reads(
                    df_reads,
                    map_start=fixed_params.get("map_start"),
                    map_end=fixed_params.get("map_end"),
                    recomb_start=fixed_params.get("recomb_start"),
                    recomb_end=fixed_params.get("recomb_end"),
                    map_col=fixed_params.get("map_col", "prefix_map"),
                    recomb_col=fixed_params.get("recomb_col", "prefix_recomb"),
                )

                df_cells = call_cells(
                    df_reads_prepped,
                    df_barcode_library=df_barcode_library,
                    q_min=q_min,
                    map_start=fixed_params.get("map_start"),
                    map_end=fixed_params.get("map_end"),
                    map_col=fixed_params.get("map_col", "prefix_map"),
                    recomb_start=fixed_params.get("recomb_start"),
                    recomb_end=fixed_params.get("recomb_end"),
                    recomb_col=fixed_params.get("recomb_col", "prefix_recomb"),
                    recomb_filter_col=fixed_params.get("recomb_filter_col", None),
                    recomb_q_thresh=fixed_params.get("recomb_q_thresh", 0.1),
                    error_correct=error_correct,
                    sort_calls=sort_calls,
                    max_distance=fixed_params.get("max_distance", 2),
                    barcode_info_cols=fixed_params.get("barcode_info_cols", None),
                )
            else:
                # Simple barcode mode
                df_cells = call_cells(
                    df_reads,
                    df_barcode_library=df_barcode_library,
                    q_min=q_min,
                    barcode_col=fixed_params.get("barcode_col", "sgRNA"),
                    prefix_col=fixed_params.get("prefix_col", None),
                    error_correct=error_correct,
                    sort_calls=sort_calls,
                    max_distance=fixed_params.get("max_distance", 2),
                )

            if df_cells is None or len(df_cells) == 0:
                results.append(
                    {
                        **current_params,
                        "metric_score": 0.0,
                        "metric_name": metric_name,
                        "total_cells": len(np.unique(mask)) - 1,
                        "status": "call_cells_failed",
                    }
                )
                continue

            # Compute metric score with total_cells as denominator
            total_cells = len(np.unique(mask)) - 1
            metric_score = metric_fn(df_cells, total_cells)

            # Annotate df_cells with parameters for tracking
            for param_name, param_value in current_params.items():
                df_cells[param_name] = param_value
            all_cells.append(df_cells)

            # Record results
            results.append(
                {
                    **current_params,
                    "metric_score": metric_score,
                    "metric_name": metric_name,
                    "total_cells": len(np.unique(mask)) - 1,
                    "status": "success",
                }
            )

        except Exception as e:
            if verbose:
                print(f"Error in combination {current_params}: {e}")
            results.append(
                {
                    **current_params,
                    "metric_score": 0.0,
                    "metric_name": metric_name,
                    "total_cells": len(np.unique(mask)) - 1,
                    "status": f"error: {str(e)[:50]}",
                }
            )

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    df_cells_combined = pd.concat(all_cells, ignore_index=True) if all_cells else None

    if verbose:
        print(f"\nCompleted {len(results_df)} combinations")
        success_count = len(results_df[results_df["status"] == "success"])
        print(f"Successful: {success_count}")
        if len(results_df[results_df["status"] != "success"]) > 0:
            print("\nFailed combinations:")
            print(results_df["status"].value_counts())

    return results_df, df_cells_combined


def metric_one_barcode_fraction(df_cells, total_cells=None):
    """Calculate fraction of cells mapping to exactly one barcode.

    A cell maps to one barcode if it has a valid gene_symbol_0 (mapped to library)
    but no gene_symbol_1 (no secondary barcode).

    This metric uses gene symbols rather than raw barcodes to ensure only
    library-matched barcodes are counted (following eval_mapping.py logic).

    **Denominator**: Uses total_cells (all segmented cells) when provided, matching
    the heatmap evaluation logic. This ensures metrics are comparable and optimization
    accounts for overall mapping success, not just purity among mapped cells.

    Parameters
    ----------
    df_cells : pd.DataFrame
        DataFrame with cell-level barcode calls from call_cells().
    total_cells : int, optional
        Total number of segmented cells (includes unmapped cells).
        If None, uses len(df_cells) as denominator (legacy behavior).

    Returns:
    -------
    float
        Fraction of cells with exactly one barcode (0.0 to 1.0).
    """
    if df_cells is None or len(df_cells) == 0:
        return 0.0

    # Check required columns exist
    if "gene_symbol_0" not in df_cells.columns:
        return 0.0

    # Count cells with valid mapped barcode_0 and no barcode_1
    # Using gene_symbol columns ensures barcodes matched the library
    has_mapped_0 = df_cells["gene_symbol_0"].notna()

    if "gene_symbol_1" not in df_cells.columns:
        num_one_barcode = has_mapped_0.sum()
    else:
        has_mapped_1 = df_cells["gene_symbol_1"].notna()
        # Count cells with exactly one mapped barcode
        has_one_barcode = has_mapped_0 & ~has_mapped_1
        num_one_barcode = has_one_barcode.sum()

    # Use total_cells as denominator if provided, otherwise len(df_cells)
    denominator = total_cells if total_cells is not None else len(df_cells)
    return num_one_barcode / denominator if denominator > 0 else 0.0


def metric_any_barcode_fraction(df_cells, total_cells=None):
    """Calculate fraction of cells mapping to any barcode(s).

    A cell maps to any barcode if it has at least one valid gene_symbol_0
    (mapped to library).

    This metric uses gene symbols rather than raw barcodes to ensure only
    library-matched barcodes are counted (following eval_mapping.py logic).

    **Denominator**: Uses total_cells (all segmented cells) when provided, matching
    the heatmap evaluation logic. This ensures metrics are comparable and optimization
    accounts for overall mapping success, not just purity among mapped cells.

    Parameters
    ----------
    df_cells : pd.DataFrame
        DataFrame with cell-level barcode calls from call_cells().
    total_cells : int, optional
        Total number of segmented cells (includes unmapped cells).
        If None, uses len(df_cells) as denominator (legacy behavior).

    Returns:
    -------
    float
        Fraction of cells with any barcode (0.0 to 1.0).
    """
    if df_cells is None or len(df_cells) == 0:
        return 0.0

    if "gene_symbol_0" not in df_cells.columns:
        return 0.0

    # Count cells with at least one valid mapped barcode
    has_any_barcode = df_cells["gene_symbol_0"].notna()
    num_any_barcode = has_any_barcode.sum()

    # Use total_cells as denominator if provided, otherwise len(df_cells)
    denominator = total_cells if total_cells is not None else len(df_cells)
    return num_any_barcode / denominator if denominator > 0 else 0.0


def metric_mapping_rate(df_cells):
    """Calculate the fraction of reads that mapped to known barcodes.

    This is calculated from cells with successful barcode calls.

    Parameters
    ----------
    df_cells : pd.DataFrame
        DataFrame with cell-level barcode calls from call_cells().

    Returns:
    -------
    float
        Fraction of cells with mapped barcodes (0.0 to 1.0).
    """
    # For now, this is equivalent to any_barcode_fraction
    # Could be extended if read-level data is available
    return metric_any_barcode_fraction(df_cells)


def metric_specificity(df_cells, total_cells=None):
    """Calculate specificity as ratio of one-barcode to all-barcode cells.

    Specificity = (cells with exactly one barcode) / (cells with any barcode)

    Interpretation:
    - 1.0 = Perfect specificity. All mapped cells have exactly one barcode.
    - 0.8 = Good specificity. 80% of mapped cells have one barcode, 20% have multiple.
    - 0.5 = Poor specificity. Half of mapped cells have multiple barcodes (likely doublets).

    **Higher is better.** Low specificity suggests doublet contamination or poor
    barcode separation. Use this metric when optimizing for single-cell purity.

    **Note**: Specificity is a ratio metric. The total_cells denominator cancels out,
    so this metric is less affected by the denominator choice than the fraction metrics.

    Parameters
    ----------
    df_cells : pd.DataFrame
        DataFrame with cell-level barcode calls from call_cells().
    total_cells : int, optional
        Total number of segmented cells (passed to fraction metrics for consistency).

    Returns:
    -------
    float
        Ratio of one-barcode cells to any-barcode cells (0.0 to 1.0).
        Returns 0.0 if no cells have any barcodes.
    """
    any_frac = metric_any_barcode_fraction(df_cells, total_cells)
    if any_frac == 0:
        return 0.0

    one_frac = metric_one_barcode_fraction(df_cells, total_cells)

    return one_frac / any_frac


def visualize_parameter_results(
    results_df,
    df_cells=None,
    metric_name="metric_score",
    show_secondary_metrics=True,
    save_plots=False,
):
    """Visualize parameter search results as heatmaps with multiple metrics.

    Creates heatmaps showing how the metric varies across parameter combinations.
    If there are 2 parameters, creates a single heatmap. If there are more,
    creates multiple heatmaps for each parameter pair.

    When df_cells is provided and show_secondary_metrics is True, displays
    a multi-panel figure with the primary metric plus standard secondary metrics:
    - fraction_one_barcode
    - fraction_any_barcode
    - specificity

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from automated_parameter_search with parameter columns
        and a metric_score column.
    df_cells : pd.DataFrame, optional
        Combined DataFrame of per-cell calls from automated_parameter_search.
        If provided and show_secondary_metrics=True, secondary metrics will
        be computed and displayed.
    metric_name : str, default='metric_score'
        Name of the column to visualize in heatmaps.
    show_secondary_metrics : bool, default=True
        If True and df_cells is provided, show multiple metric panels.
    save_plots : bool, default=False
        If True, saves plots as 'parameter_search_results.png'.

    Returns:
    -------
    pd.DataFrame
        Filtered DataFrame containing only successful parameter sets with
        computed secondary metrics (if df_cells was provided).
    """
    # Filter for successful results
    success_results = results_df[results_df["status"] == "success"].copy()

    if len(success_results) == 0:
        print("No successful results to visualize!")
        return None

    # Get the actual metric name from the results if available
    if "metric_name" in success_results.columns:
        primary_metric_name = success_results["metric_name"].iloc[0]
    else:
        # Fall back to formatting the column name
        primary_metric_name = metric_name
        if metric_name == "metric_score":
            primary_metric_name = "metric_score"

    # Compute secondary metrics if df_cells is provided
    if df_cells is not None and show_secondary_metrics:
        # Ensure df_cells has the parameter columns
        param_cols_in_cells = [
            col
            for col in success_results.columns
            if col not in [metric_name, "status", "total_cells", "metric_name"]
            and col in df_cells.columns
        ]

        if len(param_cols_in_cells) > 0:
            # Compute metrics for each parameter combination
            for idx, row in success_results.iterrows():
                # Filter df_cells for this parameter combination
                mask = pd.Series(True, index=df_cells.index)
                for param in param_cols_in_cells:
                    mask &= df_cells[param] == row[param]

                df_subset = df_cells[mask]

                # Get total_cells from results_df for this parameter combination
                total_cells_for_params = row.get("total_cells", len(df_subset))

                # Compute secondary metrics with total_cells denominator
                success_results.loc[idx, "fraction_one_barcode"] = (
                    metric_one_barcode_fraction(df_subset, total_cells_for_params)
                )
                success_results.loc[idx, "fraction_any_barcode"] = (
                    metric_any_barcode_fraction(df_subset, total_cells_for_params)
                )
                success_results.loc[idx, "specificity"] = metric_specificity(
                    df_subset, total_cells_for_params
                )

    # Identify parameter columns (exclude metric, status, total_cells, metric_name, and computed metrics)
    param_cols = [
        col
        for col in success_results.columns
        if col
        not in [
            metric_name,
            "status",
            "total_cells",
            "metric_name",
            "fraction_one_barcode",
            "fraction_any_barcode",
            "specificity",
        ]
    ]

    if len(param_cols) == 0:
        print("No parameter columns found!")
        return success_results

    # Format primary metric name for display
    display_primary_metric = primary_metric_name.replace("_", " ").title()

    # Print best result with all metrics
    best_idx = success_results[metric_name].idxmax()
    best_result = success_results.loc[best_idx]
    print(f"\nBest result ({display_primary_metric} = {best_result[metric_name]:.3f}):")
    for param in param_cols:
        print(f"  {param} = {best_result[param]}")

    # Print secondary metrics if available
    if "fraction_one_barcode" in success_results.columns:
        print(f"  fraction_one_barcode = {best_result['fraction_one_barcode']:.3f}")
    if "fraction_any_barcode" in success_results.columns:
        print(f"  fraction_any_barcode = {best_result['fraction_any_barcode']:.3f}")
    if "specificity" in success_results.columns:
        print(f"  specificity = {best_result['specificity']:.3f}")

    # Determine metrics to plot
    metrics_to_plot = [(metric_name, display_primary_metric)]
    if show_secondary_metrics and df_cells is not None:
        if "fraction_one_barcode" in success_results.columns:
            metrics_to_plot.append(("fraction_one_barcode", "Fraction One Barcode"))
        if "fraction_any_barcode" in success_results.columns:
            metrics_to_plot.append(("fraction_any_barcode", "Fraction Any Barcode"))
        if "specificity" in success_results.columns:
            metrics_to_plot.append(
                ("specificity", "Specificity (Higher=Better: 1.0=One Barcode/Cell)")
            )

    n_metrics = len(metrics_to_plot)

    # Create visualizations
    if len(param_cols) == 1:
        # Single parameter: line plots in vertical panels
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 6 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        param = param_cols[0]
        success_results_sorted = success_results.sort_values(param)

        for idx, (metric_col, display_name) in enumerate(metrics_to_plot):
            axes[idx].plot(
                success_results_sorted[param],
                success_results_sorted[metric_col],
                marker="o",
            )
            axes[idx].set_xlabel(param)
            axes[idx].set_ylabel(display_name)
            axes[idx].set_title(f"{display_name} vs {param}")
            axes[idx].grid(True, alpha=0.3)

    elif len(param_cols) == 2:
        # Two parameters: heatmaps in vertical panels
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 8 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for idx, (metric_col, display_name) in enumerate(metrics_to_plot):
            pivot = success_results.pivot_table(
                index=param_cols[0],
                columns=param_cols[1],
                values=metric_col,
                aggfunc="mean",
            )
            sns.heatmap(
                pivot,
                ax=axes[idx],
                cmap="viridis",
                annot=True,
                fmt=".3f",
                cbar_kws={"label": display_name},
            )
            axes[idx].set_title(f"{display_name} Heatmap")

    else:
        # More than 2 parameters not supported
        print(f"\nError: {len(param_cols)} parameters provided.")
        print("Automated parameter search visualization supports 1-2 parameters only.")
        print("Consider fixing some parameters or running multiple smaller searches.")
        print(
            "\nResults are still available in the returned DataFrame for manual analysis."
        )
        return success_results

    plt.tight_layout()

    if save_plots:
        plt.savefig("parameter_search_results.png", dpi=300, bbox_inches="tight")
        print("Plots saved to 'parameter_search_results.png'")

    plt.show()

    return success_results


def get_best_parameters(results_df, metric_name="metric_score", maximize=True):
    """Find the best parameter combination based on metric score.

    For built-in metrics (specificity, one_barcode_fraction, any_barcode_fraction),
    use maximize=True (default) since higher values are better. For custom metrics
    like error rate or computational cost, use maximize=False if lower is better.

    If multiple parameter sets are tied for the best score, all tied results
    will be displayed with their secondary metrics to aid in selection.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from automated_parameter_search.
    metric_name : str, default='metric_score'
        Name of the metric column to optimize.
    maximize : bool, default=True
        If True, find parameters that maximize the metric (for metrics where higher is better).
        If False, find parameters that minimize the metric (for metrics where lower is better).

    Returns:
    -------
    pd.Series
        Best parameter set with all column values.
    """
    success_results = results_df[results_df["status"] == "success"].copy()

    if len(success_results) == 0:
        print("No successful results!")
        return None

    if metric_name not in success_results.columns:
        print(f"Metric '{metric_name}' not found in results!")
        return None

    # Find best
    if maximize:
        best_idx = success_results[metric_name].idxmax()
        direction = "maximum"
    else:
        best_idx = success_results[metric_name].idxmin()
        direction = "minimum"

    best_params = success_results.loc[best_idx]
    best_score = best_params[metric_name]

    # Check for ties (using small tolerance for float comparison)
    tolerance = 1e-9
    tied_results = success_results[
        abs(success_results[metric_name] - best_score) < tolerance
    ]

    # Identify parameter columns
    param_cols = [
        col
        for col in success_results.columns
        if col
        not in [
            metric_name,
            "status",
            "total_cells",
            "metric_name",
            "fraction_one_barcode",
            "fraction_any_barcode",
            "specificity",
        ]
    ]

    # Print results
    if len(tied_results) > 1:
        print(
            f"\nWarning: {len(tied_results)} parameter sets tied for best {metric_name}!"
        )
        print(f"All tied results ({direction} {metric_name} = {best_score:.3f}):\n")

        for idx, (row_idx, row) in enumerate(tied_results.iterrows(), 1):
            print(f"Option {idx}:")
            for param in param_cols:
                print(f"  {param} = {row[param]}")
            # Show secondary metrics if available to help choose
            if "specificity" in row.index and pd.notna(row["specificity"]):
                print(f"  specificity = {row['specificity']:.3f}")
            if "fraction_one_barcode" in row.index and pd.notna(
                row["fraction_one_barcode"]
            ):
                print(f"  fraction_one_barcode = {row['fraction_one_barcode']:.3f}")
            print()

        print(
            "Suggestion: Consider secondary metrics or simpler parameter values to break the tie."
        )
    else:
        print(f"\nBest parameters ({direction} {metric_name} = {best_score:.3f}):")
        for param in param_cols:
            print(f"  {param} = {best_params[param]}")

    return best_params
