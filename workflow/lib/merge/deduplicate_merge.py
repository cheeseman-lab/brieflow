"""Functions for the deduplication step in the merge process.

This module provides utility functions to clean and deduplicate cell mappings
during the integration of phenotype and SBS datasets. It includes methods to
remove duplicate mappings and evaluate the effectiveness of the merging process.

Functions:
    deduplicate_cells: Removes duplicate cell mappings by selecting the best matches
        for phenotype and SBS cells based on quality metrics.
    check_matching_rates: Evaluates the fraction of original cells that remain after
        the deduplication and merging process.

These functions are useful in processing optical pooled screening data by ensuring
high-quality cell matching and tracking the merging efficiency.
"""

import pandas as pd


def deduplicate_cells(df, mapped_single_gene=False, return_stats=False):
    """Removes duplicate cell mappings in two steps.

    1. For each phenotype cell (`cell_0`), keeps the best SBS cell match.
    2. For each SBS cell (`cell_1`), keeps the best phenotype cell match.

    Args:
        df: DataFrame with merged cell data.
        mapped_single_gene: If True, filters to cells with single gene mappings. Defaults to False.
        return_stats: If True, returns a DataFrame with deduplication statistics. Defaults to False.

    Returns:
        If `return_stats` is False, returns a DataFrame with duplicates removed.
        If `return_stats` is True, returns a tuple of the deduplicated DataFrame and a statistics DataFrame.
    """
    # Step 1: For each phenotype cell, keep best SBS match
    # Sort by mapping quality and distance to prioritize better matches
    df_sbs_deduped = df.sort_values(
        ["mapped_single_gene", "fov_distance_1"], ascending=[False, True]
    ).drop_duplicates(["well", "tile", "cell_0"], keep="first")

    # Step 2: For each remaining SBS cell, keep best phenotype match
    df_final = df_sbs_deduped.sort_values(
        "fov_distance_0", ascending=True
    ).drop_duplicates(["well", "site", "cell_1"], keep="first")

    # Calculate statistics
    stats = {
        "stage": ["Initial", "After SBS dedup", "After phenotype dedup"],
        "total_cells": [len(df), len(df_sbs_deduped), len(df_final)],
        "mapped_genes": [
            df[df.mapped_single_gene == True].pipe(len),
            df_sbs_deduped[df_sbs_deduped.mapped_single_gene == True].pipe(len),
            df_final[df_final.mapped_single_gene == True].pipe(len),
        ],
    }

    # Print summary statistics
    print(f"Initial cells: {stats['total_cells'][0]:,}")
    print(f"After SBS deduplication: {stats['total_cells'][1]:,}")
    print(f"After phenotype deduplication: {stats['total_cells'][2]:,}")
    print(f"Final mapped genes: {stats['mapped_genes'][2]:,}")

    if mapped_single_gene:
        print("\nFilter to cells with single gene mappings.")
        df_final = df_final[df_final.mapped_single_gene]
    else:
        print("\nKeeping all deduped cells.")

    if return_stats:
        return df_final, pd.DataFrame(stats)
    else:
        return df_final


def check_matching_rates(orig_data, merged_data, modality="sbs", return_stats=False):
    """Checks what fraction of original cells survived the merging/cleaning process.

    Args:
        orig_data: Original dataset (e.g., sbs_info or phenotype_info).
        merged_data: Cleaned/merged dataset to compare against.
        modality: Either 'sbs' or 'phenotype'. Defaults to 'sbs'.
        return_stats: If True, returns a DataFrame with matching statistics. Defaults to False.

    Returns:
        DataFrame with matching statistics if `return_stats` is True.
    """
    # Set up merge parameters based on modality
    if modality == "sbs":
        merge_cols = ["well", "site", "cell_1"]
        rename_dict = {"cell": "cell_1", "tile": "site"}
    else:
        merge_cols = ["well", "tile", "cell_0"]
        rename_dict = {"label": "cell_0"}

    # Prepare and merge data
    checking_df = orig_data.rename(columns=rename_dict).merge(
        merged_data, how="left", on=merge_cols
    )

    # Calculate matching rates per well
    rates = []
    print(f"\nFinal matching rates for {modality.upper()} cells:")
    for well, df in checking_df.groupby("well"):
        total = len(df)
        matched = df.distance.notna().sum()
        rate = matched / total * 100
        print(f"Well {well}: {rate:.1f}% ({matched:,}/{total:,} cells)")

        rates.append(
            {
                "well": well,
                "total_cells": total,
                "matched_cells": matched,
                "match_rate": rate,
            }
        )

    rates_df = pd.DataFrame(rates)

    if return_stats:
        return rates_df
