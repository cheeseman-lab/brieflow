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

def deduplicate_cells(
    df, 
    mapped_single_gene=False, 
    return_stats=False,
    approach="fast",  # New parameter
    pheno_id_cols=None,  # New parameter
    sbs_id_cols=None     # New parameter
):
    """Removes duplicate cell mappings in two steps.

    1. For each phenotype cell, keeps the best SBS cell match.
    2. For each SBS cell, keeps the best phenotype cell match.

    Args:
        df: DataFrame with merged cell data.
        mapped_single_gene: If True, filters to cells with single gene mappings. Defaults to False.
        return_stats: If True, returns a DataFrame with deduplication statistics. Defaults to False.
        approach: "fast" or "stitch" to determine deduplication strategy. Defaults to "fast".
        pheno_id_cols: Column(s) identifying phenotype cells. Auto-determined if None.
        sbs_id_cols: Column(s) identifying SBS cells. Auto-determined if None.

    Returns:
        If `return_stats` is False, returns a DataFrame with duplicates removed.
        If `return_stats` is True, returns a tuple of the deduplicated DataFrame and a statistics DataFrame.
    """
    
    # Auto-determine ID columns based on approach
    if pheno_id_cols is None or sbs_id_cols is None:
        if approach == "stitch":
            pheno_id_cols = "stitched_cell_id_0"
            sbs_id_cols = "stitched_cell_id_1"
        else:  # fast approach
            pheno_id_cols = ["plate", "well", "tile", "cell_0"]
            sbs_id_cols = ["plate", "well", "site", "cell_1"]
    
    # Step 1: For each phenotype cell, keep best SBS match
    if approach == "stitch":
        # For stitched approach, sort by distance only
        df_sbs_deduped = df.sort_values(
            "distance", ascending=True
        ).drop_duplicates(pheno_id_cols, keep="first")
    else:
        # Keep original tile approach logic
        df_sbs_deduped = df.sort_values(
            ["mapped_single_gene", "fov_distance_1"], ascending=[False, True]
        ).drop_duplicates(pheno_id_cols, keep="first")

    # Step 2: For each remaining SBS cell, keep best phenotype match
    if approach == "stitch":
        # For stitched approach, sort by distance only
        df_final = df_sbs_deduped.sort_values(
            "distance", ascending=True
        ).drop_duplicates(sbs_id_cols, keep="first")
    else:
        # Keep original tile approach logic
        df_final = df_sbs_deduped.sort_values(
            "fov_distance_0", ascending=True
        ).drop_duplicates(sbs_id_cols, keep="first")

    # Rest of the function stays exactly the same...
    stats = {
        "stage": ["Initial", "After SBS dedup", "After phenotype dedup"],
        "total_cells": [len(df), len(df_sbs_deduped), len(df_final)],
        "mapped_genes": [
            df[df.mapped_single_gene == True].pipe(len),
            df_sbs_deduped[df_sbs_deduped.mapped_single_gene == True].pipe(len),
            df_final[df_final.mapped_single_gene == True].pipe(len),
        ],
    }

    # Print summary statistics (same as before)
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

def analyze_distance_distribution(df):
    """Analyzes distance distribution for validation.
    
    Args:
        df: DataFrame with 'distance' column
        
    Returns:
        Dictionary with distance distribution metrics
    """
    if df.empty or 'distance' not in df.columns:
        return {}
        
    distances = df['distance']
    
    return {
        "distance_stats": {
            "mean": float(distances.mean()),
            "median": float(distances.median()),
            "std": float(distances.std()),
            "min": float(distances.min()),
            "max": float(distances.max()),
            "p95": float(distances.quantile(0.95)),
        },
        "distance_distribution": {
            "under_1px": int((distances < 1).sum()),
            "under_2px": int((distances < 2).sum()),
            "under_5px": int((distances < 5).sum()),
            "under_10px": int((distances < 10).sum()),
            "over_20px": int((distances > 20).sum()),
            "over_50px": int((distances > 50).sum()),
        }
    }