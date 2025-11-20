"""Cell deduplication utilities for optical pooled screening merge workflows.

This module provides functions to deduplicate cell mappings between phenotype
and sequencing-by-synthesis (SBS) datasets using configurable quality metrics.

Functions:
    deduplicate_cells: Two-step deduplication using customizable sorting priorities
    check_matching_rates: Quantifies cell retention after merge and deduplication
    analyze_distance_distribution: Evaluates spatial alignment quality metrics
"""

import pandas as pd


def deduplicate_cells(
    df,
    mapped_single_gene=False,
    return_stats=False,
    approach="fast",
    pheno_id_cols=None,
    sbs_id_cols=None,
    sbs_dedup_prior=None,
    pheno_dedup_prior=None,
):
    """Remove duplicate cell mappings through two-step deduplication process.

    Performs sequential deduplication to ensure one-to-one cell correspondence:
    1. For each phenotype cell, retain the highest-quality SBS match
    2. For each remaining SBS cell, retain the highest-quality phenotype match

    Parameters
    ----------
    df : pandas.DataFrame
        Merged cell data containing both phenotype and SBS cell information
    mapped_single_gene : bool, default False
        Filter output to cells with unambiguous single gene mappings
    return_stats : bool, default False
        Return deduplication statistics alongside deduplicated data
    approach : {"fast", "stitch"}, default "fast"
        Determines cell identification column strategy
    pheno_id_cols : str or list of str, optional
        Column(s) uniquely identifying phenotype cells. Auto-determined if None
    sbs_id_cols : str or list of str, optional
        Column(s) uniquely identifying SBS cells. Auto-determined if None
    sbs_dedup_prior : dict
        Sorting priorities for step 1. Keys are column names, values are ascending
        sort order (True/False). Required parameter - see examples below
    pheno_dedup_prior : dict
        Sorting priorities for step 2. Required parameter - see examples below

    Returns:
    -------
    pandas.DataFrame or tuple
        Deduplicated cell mappings. If return_stats=True, returns tuple of
        (deduplicated_data, statistics_dataframe)

    Raises:
    ------
    ValueError
        If sbs_dedup_prior or pheno_dedup_prior is not specified

    """
    # Determine cell identification columns based on merge approach
    if pheno_id_cols is None or sbs_id_cols is None:
        if approach == "stitch":
            pheno_id_cols = "stitched_cell_id_0"
            sbs_id_cols = "stitched_cell_id_1"
        else:  # fast approach
            pheno_id_cols = ["plate", "well", "tile", "cell_0"]
            sbs_id_cols = ["plate", "well", "site", "cell_1"]

    # Validate required deduplication priorities
    if sbs_dedup_prior is None:
        raise ValueError(
            "sbs_dedup_prior must be specified. See documentation for examples."
        )

    if pheno_dedup_prior is None:
        raise ValueError(
            "pheno_dedup_prior must be specified. See documentation for examples."
        )

    # Step 1: Retain best SBS match for each phenotype cell
    df_sbs_deduped = df.sort_values(
        list(sbs_dedup_prior.keys()), ascending=list(sbs_dedup_prior.values())
    ).drop_duplicates(pheno_id_cols, keep="first")

    # Step 2: Retain best phenotype match for each remaining SBS cell
    df_final = df_sbs_deduped.sort_values(
        list(pheno_dedup_prior.keys()), ascending=list(pheno_dedup_prior.values())
    ).drop_duplicates(sbs_id_cols, keep="first")

    # Compile deduplication statistics
    stats = {
        "stage": ["Initial", "After SBS dedup", "After phenotype dedup"],
        "total_cells": [len(df), len(df_sbs_deduped), len(df_final)],
        "mapped_genes": [
            df[df.mapped_single_gene == True].pipe(len),
            df_sbs_deduped[df_sbs_deduped.mapped_single_gene == True].pipe(len),
            df_final[df_final.mapped_single_gene == True].pipe(len),
        ],
    }

    # Display deduplication summary
    print(f"Initial cells: {stats['total_cells'][0]:,}")
    print(f"After SBS deduplication: {stats['total_cells'][1]:,}")
    print(f"After phenotype deduplication: {stats['total_cells'][2]:,}")
    print(f"Final mapped genes: {stats['mapped_genes'][2]:,}")

    # Apply single gene mapping filter if requested
    if mapped_single_gene:
        print("\nFiltering to cells with single gene mappings.")
        df_final = df_final[df_final.mapped_single_gene]
    else:
        print("\nRetaining all deduplicated cells.")

    return (df_final, pd.DataFrame(stats)) if return_stats else df_final


def check_matching_rates(orig_data, merged_data, modality="sbs", return_stats=False):
    """Calculate cell retention rates after merge and deduplication processing.

    Quantifies the fraction of original cells that successfully matched and
    survived the merge/deduplication pipeline.

    Parameters
    ----------
    orig_data : pandas.DataFrame
        Original unprocessed dataset (sbs_info or phenotype_info)
    merged_data : pandas.DataFrame
        Final processed dataset after merge and deduplication
    modality : {"sbs", "phenotype"}, default "sbs"
        Data modality being evaluated, determines column mapping strategy
    return_stats : bool, default False
        Return detailed matching statistics DataFrame

    Returns:
    -------
    pandas.DataFrame, optional
        Per-well matching statistics if return_stats=True. Contains columns:
        well, total_cells, matched_cells, match_rate

    """
    # Configure merge parameters based on data modality
    if modality == "sbs":
        merge_cols = ["well", "site", "cell_1"]
        rename_dict = {"cell": "cell_1", "tile": "site"}
    else:
        merge_cols = ["well", "tile", "cell_0"]
        rename_dict = {"label": "cell_0"}

    # Harmonize column names and identify matched cells
    checking_df = orig_data.rename(columns=rename_dict).merge(
        merged_data, how="left", on=merge_cols
    )

    # Calculate per-well matching statistics
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

    return pd.DataFrame(rates) if return_stats else None


def analyze_distance_distribution(df):
    """Analyze distance distribution for spatial alignment quality assessment.

    Computes distance statistics and distribution metrics to assess the quality
    of cell-to-cell spatial alignments in merged datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged dataset containing 'distance' column with spatial alignment distances

    Returns:
    -------
    dict
        Dictionary containing:
        - distance_stats: Statistical summary (mean, median, std, min, max, p95)
        - distance_distribution: Counts at various distance thresholds

    """
    if df.empty or "distance" not in df.columns:
        return {}

    distances = df["distance"]

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
        },
    }
