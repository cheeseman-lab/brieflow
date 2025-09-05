"""Module for automating parameter search in SBS.

This module provides functions to perform an automated parameter search.
It tests different combinations of peak width and threshold reads,
and evaluates the results based on the fraction of cells mapping to barcodes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def automated_parameter_search(
    aligned_images,
    nuclei_mask,
    barcodes,
    df_barcode_library,
    wildcards,
    bases,
    extra_channel_indices,
    max_filter_width=3,
    peak_width_range=[5, 10, 15, 20],
    threshold_range=[50, 100, 150, 200, 250],
    call_reads_method="percentile",
    map_start=None,
    map_end=None,
    recomb_start=None,
    recomb_end=None,
    map_col="prefix_map",
    recomb_col="prefix_recomb",
    recomb_filter_col="Q_recomb",
    q_min=0,
    error_correct=True,
    recomb_q_thresh=0.5,
    sort_calls="peak",
    verbose=False,
):
    """Automated parameter search for SBS processing.

    This function systematically tests combinations of `peak_width` and `threshold_reads`
    to optimize SBS (sequencing by synthesis) image analysis. It returns a summary DataFrame
    of results for each parameter combination and a combined DataFrame of per-cell calls.

    Parameters
    ----------
    aligned_images : np.ndarray
        Array of aligned SBS images (channels x height x width).
    nuclei_mask : np.ndarray
        Labeled mask array where each nucleus has a unique integer label.
    barcodes : list or np.ndarray
        List of barcode sequences to map reads against.
    df_barcode_library : pd.DataFrame
        DataFrame containing barcode library information for mapping.
    wildcards : dict
        Dictionary of wildcards for barcode extraction (e.g., {'N': '[ACGT]'}).
    bases : list
        List of base labels (e.g., ['A', 'C', 'G', 'T']).
    extra_channel_indices : list or np.ndarray
        Indices of channels to skip during filtering (e.g., DAPI or unused channels).
    max_filter_width : int, default=3
        Width of the max filter applied to images.
    peak_width_range : list, default=[5, 10, 15, 20]
        List of peak widths to test for peak detection.
    threshold_range : list, default=[50, 100, 150, 200, 250]
        List of read count thresholds to test for base calling.
    call_reads_method : str, default="percentile"
        Method for calling reads from base intensities ('percentile', 'threshold', etc.).
    map_start : int or None, optional
        Start position for mapping region in barcode sequence.
    map_end : int or None, optional
        End position for mapping region in barcode sequence.
    recomb_start : int or None, optional
        Start position for recombination region in barcode sequence.
    recomb_end : int or None, optional
        End position for recombination region in barcode sequence.
    map_col : str, default='prefix_map'
        Column name in df_reads for mapping region.
    recomb_col : str, default='prefix_recomb'
        Column name in df_reads for recombination region.
    recomb_filter_col : str, default='Q_recomb'
        Column name for recombination quality filter.
    q_min : float, default=0
        Minimum quality threshold for mapping.
    error_correct : bool, default=True
        Whether to perform error correction during mapping.
    recomb_q_thresh : float, default=0.5
        Minimum recombination quality threshold.
    sort_calls : str, default="peak"
        Method for sorting base calls ('peak', etc.).
    verbose : bool, default=False
        If True, print progress and debug information.

    Returns:
    -------
    results_df : pd.DataFrame
        DataFrame summarizing results for each parameter combination.
    df_cells_combined : pd.DataFrame or None
        Combined DataFrame of per-cell calls for all successful parameter sets.
    """
    try:
        from lib.shared.log_filter import log_filter
        from lib.sbs.max_filter import max_filter
        from lib.sbs.compute_standard_deviation import compute_standard_deviation
        from lib.sbs.find_peaks import find_peaks
        from lib.sbs.extract_bases import extract_bases
        from lib.sbs.call_reads import call_reads
        from lib.sbs.call_cells_multi import prep_multi_reads, call_cells_multi
        from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal
        from lib.sbs.eval_mapping import plot_cell_mapping_heatmap
    except ImportError as e:
        print(f"Error importing required functions: {e}")
        return pd.DataFrame(), None

    results = []
    all_cells = []

    if verbose:
        print("Pre-processing images...")

    try:
        loged = log_filter(aligned_images, skip_index=extra_channel_indices)
        maxed = max_filter(
            loged, width=max_filter_width, remove_index=extra_channel_indices
        )

        df_sbs_info = extract_phenotype_minimal(
            phenotype_data=nuclei_mask, nuclei_data=nuclei_mask, wildcards=wildcards
        )
    except Exception as e:
        if verbose:
            print(f"Pre-processing failed: {e}")
        return pd.DataFrame(), None

    total_combinations = len(peak_width_range) * len(threshold_range)
    current_combo = 0

    for peak_width in peak_width_range:
        try:
            standard_deviation = compute_standard_deviation(
                loged, remove_index=extra_channel_indices
            )
            peaks = find_peaks(standard_deviation, width=peak_width)

            if peaks is None or (hasattr(peaks, "__len__") and len(peaks) == 0):
                for threshold_reads in threshold_range:
                    current_combo += 1
                    results.append(
                        {
                            "peak_width": peak_width,
                            "threshold_reads": threshold_reads,
                            "total_reads": 0,
                            "fraction_one_barcode": 0.0,
                            "fraction_any_barcode": 0.0,
                            "cells_with_one_barcode": 0,
                            "cells_with_any_barcode": 0,
                            "total_cells": len(np.unique(nuclei_mask)) - 1,
                            "status": "no_peaks",
                        }
                    )
                    if verbose and current_combo % 5 == 0:
                        print(f"Progress: {current_combo}/{total_combinations}")
                continue

        except Exception:
            for threshold_reads in threshold_range:
                current_combo += 1
                results.append(
                    {
                        "peak_width": peak_width,
                        "threshold_reads": threshold_reads,
                        "total_reads": 0,
                        "fraction_one_barcode": 0.0,
                        "fraction_any_barcode": 0.0,
                        "cells_with_one_barcode": 0,
                        "cells_with_any_barcode": 0,
                        "total_cells": len(np.unique(nuclei_mask)) - 1,
                        "status": "peak_error",
                    }
                )
            continue

        for threshold_reads in threshold_range:
            current_combo += 1
            if verbose and current_combo % 5 == 0:
                print(f"Progress: {current_combo}/{total_combinations}")

            try:
                df_bases = extract_bases(
                    peaks,
                    maxed,
                    nuclei_mask,
                    threshold_reads,
                    wildcards=wildcards,
                    bases=bases,
                )
                if df_bases is None or len(df_bases) == 0:
                    results.append(
                        {
                            "peak_width": peak_width,
                            "threshold_reads": threshold_reads,
                            "total_reads": 0,
                            "fraction_one_barcode": 0.0,
                            "fraction_any_barcode": 0.0,
                            "cells_with_one_barcode": 0,
                            "cells_with_any_barcode": 0,
                            "total_cells": len(np.unique(nuclei_mask)) - 1,
                            "status": "no_bases",
                        }
                    )
                    continue

                df_reads = call_reads(
                    df_bases, peaks_data=peaks, method=call_reads_method
                )
                if df_reads is None or len(df_reads) == 0:
                    results.append(
                        {
                            "peak_width": peak_width,
                            "threshold_reads": threshold_reads,
                            "total_reads": 0,
                            "fraction_one_barcode": 0.0,
                            "fraction_any_barcode": 0.0,
                            "cells_with_one_barcode": 0,
                            "cells_with_any_barcode": 0,
                            "total_cells": len(np.unique(nuclei_mask)) - 1,
                            "status": "no_reads",
                        }
                    )
                    continue

                df_reads_multi = prep_multi_reads(
                    df_reads,
                    map_start=map_start,
                    map_end=map_end,
                    recomb_start=recomb_start,
                    recomb_end=recomb_end,
                    map_col=map_col,
                    recomb_col=recomb_col,
                )

                if df_reads_multi is None or len(df_reads_multi) == 0:
                    results.append(
                        {
                            "peak_width": peak_width,
                            "threshold_reads": threshold_reads,
                            "total_reads": len(df_reads),
                            "fraction_one_barcode": 0.0,
                            "fraction_any_barcode": 0.0,
                            "cells_with_one_barcode": 0,
                            "cells_with_any_barcode": 0,
                            "total_cells": len(np.unique(nuclei_mask)) - 1,
                            "status": "prep_multi_failed",
                        }
                    )
                    continue

                df_cells = call_cells_multi(
                    df_reads_multi,
                    df_barcode_library=df_barcode_library,
                    q_min=q_min,
                    map_col=map_col,
                    error_correct=error_correct,
                    recomb_col=recomb_col,
                    recomb_filter_col=recomb_filter_col,
                    recomb_q_thresh=recomb_q_thresh,
                )

                if df_cells is None or len(df_cells) == 0:
                    results.append(
                        {
                            "peak_width": peak_width,
                            "threshold_reads": threshold_reads,
                            "total_reads": len(df_reads),
                            "fraction_one_barcode": 0.0,
                            "fraction_any_barcode": 0.0,
                            "cells_with_one_barcode": 0,
                            "cells_with_any_barcode": 0,
                            "total_cells": len(np.unique(nuclei_mask)) - 1,
                            "status": "call_cells_failed",
                        }
                    )
                    continue

                # Annotate for later tracking
                df_cells["peak_width"] = peak_width
                df_cells["threshold_reads"] = threshold_reads
                all_cells.append(df_cells)

                def get_mapping_fraction(summary_df):
                    if summary_df is None or len(summary_df) == 0:
                        return 0.0, 0
                    fraction_cols = [
                        col for col in summary_df.columns if "fraction" in col.lower()
                    ]
                    if len(fraction_cols) == 0:
                        return 0.0, 0
                    fraction = float(summary_df[fraction_cols[0]].iloc[0])
                    total_cells = len(np.unique(nuclei_mask)) - 1
                    cells = int(fraction * total_cells)
                    return fraction, cells

                try:
                    one_barcode_summary = plot_cell_mapping_heatmap(
                        df_cells,
                        df_sbs_info,
                        barcodes,
                        mapping_to="one",
                        mapping_strategy="gene symbols",
                        shape="6W_sbs",
                        return_plot=False,
                        return_summary=True,
                    )
                except Exception:
                    one_barcode_summary = None

                try:
                    any_barcode_summary = plot_cell_mapping_heatmap(
                        df_cells,
                        df_sbs_info,
                        barcodes,
                        mapping_to="any",
                        mapping_strategy="gene symbols",
                        shape="6W_sbs",
                        return_plot=False,
                        return_summary=True,
                    )
                except Exception:
                    any_barcode_summary = None

                fraction_one, cells_one = get_mapping_fraction(one_barcode_summary)
                fraction_any, cells_any = get_mapping_fraction(any_barcode_summary)
                total_cells = len(np.unique(nuclei_mask)) - 1

                results.append(
                    {
                        "peak_width": peak_width,
                        "threshold_reads": threshold_reads,
                        "total_reads": len(df_reads),
                        "fraction_one_barcode": fraction_one,
                        "fraction_any_barcode": fraction_any,
                        "cells_with_one_barcode": cells_one,
                        "cells_with_any_barcode": cells_any,
                        "total_cells": total_cells,
                        "status": "success",
                    }
                )

            except Exception:
                results.append(
                    {
                        "peak_width": peak_width,
                        "threshold_reads": threshold_reads,
                        "total_reads": 0,
                        "fraction_one_barcode": 0.0,
                        "fraction_any_barcode": 0.0,
                        "cells_with_one_barcode": 0,
                        "cells_with_any_barcode": 0,
                        "total_cells": len(np.unique(nuclei_mask)) - 1,
                        "status": "error",
                    }
                )

    results_df = pd.DataFrame(results)
    df_cells_combined = pd.concat(all_cells, ignore_index=True) if all_cells else None

    if verbose:
        print(f"\nCompleted {len(results_df)} combinations")
        print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")
        if len(results_df[results_df["status"] != "success"]) > 0:
            print("Failed combinations:")
            print(results_df["status"].value_counts())

    return results_df, df_cells_combined


def compute_peak_fraction(df_cells):
    """Compute fraction of cells where peak_1 >= 0.1 * peak_0 among all valid peak_0 cells, treating NaN peak_1 as not significant.

    Parameters
    ----------
    df_cells : pd.DataFrame
        Must include 'peak_0', 'peak_1', 'peak_width', 'threshold_reads'

    Returns:
    -------
    pd.DataFrame with:
        - peak_width
        - threshold_reads
        - fraction_peak1_significant
        - n_cells (total cells with valid peak_0)
    """
    if df_cells is None or len(df_cells) == 0:
        return pd.DataFrame(
            columns=["peak_width", "threshold_reads", "fraction_peak1_significant"]
        )

    required_cols = ["peak_0", "peak_1", "peak_width", "threshold_reads"]
    missing_cols = [col for col in required_cols if col not in df_cells.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(
            columns=["peak_width", "threshold_reads", "fraction_peak1_significant"]
        )

    df = df_cells.copy()

    # Keep only rows where peak_0 is valid (>0)
    df = df[df["peak_0"].notna() & (df["peak_0"] > 0)]

    if df.empty:
        return pd.DataFrame(
            columns=["peak_width", "threshold_reads", "fraction_peak1_significant"]
        )

    # Treat NaN peak_1 as not significant by filling with 0 (fails the >= 0.1 test)
    df["peak_1_filled"] = df["peak_1"].fillna(0)

    # Mark significant if peak_1 >= 10% of peak_0
    df["significant_peak1"] = df["peak_1_filled"] >= 0.1 * df["peak_0"]

    # Group and compute fraction
    summary = (
        df.groupby(["peak_width", "threshold_reads"])["significant_peak1"]
        .agg(fraction_peak1_significant="mean", n_cells="count")
        .reset_index()
    )

    return summary


def visualize_parameter_results(results_df, df_cells=None, save_plots=False):
    """Visualize the results of parameter search for barcode mapping performance.

    This function generates a 2x2 grid of heatmaps summarizing key metrics:
    fraction of cells mapped to any barcode, the fraction mapped to a single barcode,
    the gap between these fractions, and the significance of a secondary peak in the data.
    If cell-level data is provided, additional metrics related to peak significance are computed and visualized.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing parameter search results. Must include columns:
        'status', 'peak_width', 'threshold_reads', 'fraction_any_barcode',
        'fraction_one_barcode', and optionally 'total_reads'.
    df_cells : pandas.DataFrame, optional
        DataFrame with cell-level data for computing peak significance metrics.
        If provided, must be compatible with `compute_peak_fraction`.
    save_plots : bool, default False
        If True, saves the generated plots as 'parameter_search_results.png' in the current directory.

    Returns:
    -------
    pandas.DataFrame or None
        Filtered DataFrame of successful parameter sets, possibly augmented with peak significance metrics.
        Returns None if there are no successful results to visualize.

    Notes:
    -----
    - Only parameter sets with status 'success' are visualized.
    - If `df_cells` is provided, the function computes and displays the top parameter sets by peak1 significance.
    - The function uses seaborn and matplotlib for plotting.
    - The four heatmaps include:
        1. Fraction mapping to any barcode.
        2. Fraction mapping to one barcode.
        3. Gap between 'any' and 'one' mapping fractions.
        4. Fraction of cells with significant peak1 (if available), or total reads otherwise.
    """
    # Filter for successful results
    success_results = results_df[results_df["status"] == "success"].copy()

    if len(success_results) == 0:
        print("No successful results to visualize!")
        return None

    # Compute peak fraction if df_cells provided
    if df_cells is not None:
        peak_fractions = compute_peak_fraction(df_cells)
        if len(peak_fractions) > 0:
            success_results = success_results.merge(
                peak_fractions, on=["peak_width", "threshold_reads"], how="left"
            )
            print("\nTop parameter sets by peak1 significance:")
            if "fraction_peak1_significant" in success_results.columns:
                top_peak = success_results.nlargest(5, "fraction_peak1_significant")[
                    [
                        "peak_width",
                        "threshold_reads",
                        "fraction_peak1_significant",
                        "n_cells",
                    ]
                ]
                print(top_peak.to_string(index=False, float_format="{:.3f}".format))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Fraction mapping to any barcode
    pivot_any = success_results.pivot_table(
        index="peak_width",
        columns="threshold_reads",
        values="fraction_any_barcode",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot_any,
        ax=axes[0, 0],
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Fraction"},
    )
    axes[0, 0].set_title("Fraction Mapping to Any Barcode")

    # Plot 2: Gap between any and one
    success_results["mapping_gap"] = (
        success_results["fraction_any_barcode"]
        - success_results["fraction_one_barcode"]
    )
    pivot_gap = success_results.pivot_table(
        index="peak_width",
        columns="threshold_reads",
        values="mapping_gap",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot_gap,
        ax=axes[1, 1],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Gap"},
    )
    axes[1, 1].set_title("Mapping Gap (Any - One)")

    # Plot 3: Fraction mapping to one barcode
    pivot_one = success_results.pivot_table(
        index="peak_width",
        columns="threshold_reads",
        values="fraction_one_barcode",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot_one,
        ax=axes[1, 0],
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Fraction"},
    )
    axes[1, 0].set_title("Fraction Mapping to One Barcode")

    # Plot 4: Peak1 significance (if available)
    if "fraction_peak1_significant" in success_results.columns:
        pivot_peak = success_results.pivot_table(
            index="peak_width",
            columns="threshold_reads",
            values="fraction_peak1_significant",
            aggfunc="mean",
        )
        sns.heatmap(
            pivot_peak,
            ax=axes[0, 1],
            cmap="magma",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Fraction"},
        )
        axes[0, 1].set_title(
            "Cells with Significant Peak1\n(Intensity of Peak1 ≥ 10% Peak0)"
        )
    else:
        # Plot total reads instead
        pivot_reads = success_results.pivot_table(
            index="peak_width",
            columns="threshold_reads",
            values="total_reads",
            aggfunc="mean",
        )
        sns.heatmap(
            pivot_reads,
            ax=axes[1, 1],
            cmap="plasma",
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Count"},
        )
        axes[1, 1].set_title("Total Reads")

    plt.tight_layout()

    if save_plots:
        plt.savefig("parameter_search_results.png", dpi=300, bbox_inches="tight")
        print("Plots saved to 'parameter_search_results.png'")

    plt.show()

    return success_results


def get_best_parameters(results_df, df_cells=None, priority="balanced"):
    """Get recommended parameters based on priority.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with parameter search results.
    df_cells : pd.DataFrame or None
        Combined df_cells used to calculate peak1 significance.
    priority : str
        - "accuracy": lowest 'fraction_peak1_significant' with highest 'fraction_any_barcode'
        - "single_peak": highest 'fraction_one_barcode'
        - "balanced": highest 'fraction_any_barcode' with 'fraction_peak1_significant' <= 0.05
    """
    success_results = results_df[results_df["status"] == "success"].copy()

    if len(success_results) == 0:
        print("No successful results!")
        return None

    # Merge in peak1 significance if requested
    if priority in ("accuracy", "balanced"):
        if df_cells is None:
            print(
                "Warning: df_cells is required for 'accuracy' or 'balanced' but not provided."
            )
            return None

        peak_fractions = compute_peak_fraction(df_cells)
        if len(peak_fractions) == 0:
            print("Warning: peak fraction computation returned no results.")
            return None

        success_results = success_results.merge(
            peak_fractions, on=["peak_width", "threshold_reads"], how="left"
        )

    if priority == "accuracy":
        sorted_df = success_results.sort_values(
            by=["fraction_peak1_significant", "fraction_any_barcode"],
            ascending=[True, False],
        )
        best_idx = sorted_df.index[0]
        metric = "lowest peak1 significance with high any-barcode mapping"

    elif priority == "single_peak":
        best_idx = success_results["fraction_one_barcode"].idxmax()
        metric = "highest one-barcode mapping"

    elif priority == "balanced":
        filtered = success_results[
            success_results["fraction_peak1_significant"] <= 0.05
        ]
        if filtered.empty:
            print(
                "No results with peak1 significance ≤ 0.05! Returning best overall any-barcode mapping."
            )
            best_idx = success_results["fraction_any_barcode"].idxmax()
            metric = "any-barcode mapping (fallback)"
        else:
            best_idx = filtered["fraction_any_barcode"].idxmax()
            metric = "balanced (high any-barcode with low peak1 significance)"

    else:
        raise ValueError(f"Unknown priority setting: {priority}")

    best_params = success_results.loc[best_idx]

    print(f"\nBest parameters for {metric}:")
    print(f"  PEAK_WIDTH = {int(best_params['peak_width'])}")
    print(f"  THRESHOLD_READS = {int(best_params['threshold_reads'])}")
    print(f"  One-barcode mapping: {best_params['fraction_one_barcode']:.3f}")
    print(f"  Any-barcode mapping: {best_params['fraction_any_barcode']:.3f}")
    if "fraction_peak1_significant" in best_params:
        print(f"  Peak1 significance: {best_params['fraction_peak1_significant']:.3f}")
    print(f"  Total reads: {int(best_params['total_reads'])}")

    return best_params
