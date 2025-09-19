"""Well Deduplication Library.

This module provides spatial deduplication functionality for cell matching data.
It focuses on stitched cell ID deduplication with comprehensive validation,
ensuring 1:1 spatial mapping while maintaining legacy compatibility through
preservation of original cell IDs.

Key Functions:
    - legacy_deduplication_stitched_ids: Core deduplication using stitched IDs
    - validate_final_matches: Comprehensive validation with quality metrics
    - analyze_duplicates: Pattern analysis for any ID columns
    - get_quality_summary: Concise quality metrics for reporting

The deduplication process:
1. For each phenotype stitched cell, keeps the best SBS match (lowest distance)
2. For each SBS stitched cell, keeps the best phenotype match (lowest distance)
3. Preserves original cell IDs for downstream compatibility
4. Provides detailed quality and validation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def validate_final_matches(final_matches: pd.DataFrame) -> Dict[str, Any]:
    """Validate final matches for spatial accuracy and quality.

    Performs comprehensive validation of deduplicated cell matches, checking for:
    - 1:1 mapping in stitched cell IDs (spatial accuracy requirement)
    - Distance distribution and quality metrics
    - Duplicate detection in both stitched and original IDs
    - Overall match quality assessment

    Args:
        final_matches: DataFrame containing final cell matches with columns:
            - stitched_cell_id_0, stitched_cell_id_1: Stitched cell identifiers
            - cell_0, cell_1: Original cell identifiers
            - distance: Euclidean distance between matched cells

    Returns:
        Dictionary containing comprehensive validation results:
            - status: 'empty' or 'valid'
            - match_count: Number of final matches
            - is_1to1_stitched: Whether stitched IDs have 1:1 mapping
            - has_stitched_ids: Whether stitched ID columns are present
            - stitched_duplicates: Count of duplicates in stitched IDs
            - original_duplicates: Count of duplicates in original IDs
            - distance_stats: Statistical measures of match distances
            - distance_distribution: Counts in various distance ranges
            - quality_metrics: Precision metrics and overall quality tier
            - validation_status: Overall validation result
            - issues_detected: Boolean flags for various issues

    Example:
        >>> matches = pd.DataFrame({
        ...     'stitched_cell_id_0': [1, 2, 3],
        ...     'stitched_cell_id_1': [101, 102, 103],
        ...     'cell_0': [1, 2, 3],
        ...     'cell_1': [101, 102, 103],
        ...     'distance': [1.5, 2.1, 0.8]
        ... })
        >>> result = validate_final_matches(matches)
        >>> result['is_1to1_stitched']
        True
        >>> result['quality_metrics']['quality_tier']
        'excellent'
    """
    if final_matches.empty:
        return {
            "status": "empty",
            "match_count": 0,
            "is_1to1_stitched": True,  # Technically true for empty set
            "validation_status": "empty",
        }

    # Check for stitched ID duplicates (should be 1:1 for spatial accuracy)
    has_stitched_ids = (
        "stitched_cell_id_0" in final_matches.columns
        and "stitched_cell_id_1" in final_matches.columns
    )

    if has_stitched_ids:
        stitched_pheno_dups = final_matches["stitched_cell_id_0"].duplicated().sum()
        stitched_sbs_dups = final_matches["stitched_cell_id_1"].duplicated().sum()
        is_1to1_stitched = stitched_pheno_dups == 0 and stitched_sbs_dups == 0
    else:
        stitched_pheno_dups = 0
        stitched_sbs_dups = 0
        is_1to1_stitched = False

    # Check original ID duplicates (may exist - acceptable for legacy compatibility)
    original_pheno_dups = final_matches["cell_0"].duplicated().sum()
    original_sbs_dups = final_matches["cell_1"].duplicated().sum()

    # Distance statistics
    distances = final_matches["distance"]

    # Quality distribution - count matches in various distance ranges
    under_1px = (distances < 1).sum()
    under_2px = (distances < 2).sum()
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_20px = (distances > 20).sum()
    over_50px = (distances > 50).sum()

    # Overall quality assessment based on precision and distance metrics
    precision_5px = under_5px / len(distances)
    mean_dist = distances.mean()

    if mean_dist < 2.0 and precision_5px > 0.8 and over_50px == 0:
        quality_tier = "excellent"
    elif mean_dist < 5.0 and precision_5px > 0.6 and over_50px == 0:
        quality_tier = "good"
    elif over_50px == 0:
        quality_tier = "acceptable"
    else:
        quality_tier = "poor"

    return {
        "status": "valid",
        "match_count": len(final_matches),
        "is_1to1_stitched": is_1to1_stitched,
        "has_stitched_ids": has_stitched_ids,
        "stitched_duplicates": {
            "phenotype": int(stitched_pheno_dups),
            "sbs": int(stitched_sbs_dups),
        },
        "original_duplicates": {
            "phenotype": int(original_pheno_dups),
            "sbs": int(original_sbs_dups),
        },
        "distance_stats": {
            "mean": float(distances.mean()),
            "median": float(distances.median()),
            "std": float(distances.std()),
            "min": float(distances.min()),
            "max": float(distances.max()),
            "p95": float(distances.quantile(0.95)),
        },
        "distance_distribution": {
            "under_1px": int(under_1px),
            "under_2px": int(under_2px),
            "under_5px": int(under_5px),
            "under_10px": int(under_10px),
            "over_20px": int(over_20px),
            "over_50px": int(over_50px),
        },
        "quality_metrics": {
            "precision_1px": float(under_1px / len(distances)),
            "precision_2px": float(under_2px / len(distances)),
            "precision_5px": float(precision_5px),
            "precision_10px": float(under_10px / len(distances)),
            "quality_tier": quality_tier,
        },
        "validation_status": "valid"
        if is_1to1_stitched
        else "stitched_duplicates_present",
        "issues_detected": {
            "large_distances": over_50px > 0,
            "stitched_duplicates": not is_1to1_stitched and has_stitched_ids,
            "no_stitched_ids": not has_stitched_ids,
        },
    }


def deduplicate_matches_by_stitched_ids(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate cell matches using stitched cell IDs for accurate 1:1 cell mapping.

    This is the core deduplication function that ensures 1:1 spatial mapping
    between phenotype and SBS cells using stitched cell identifiers for
    maximum spatial accuracy, while preserving original cell IDs for
    downstream processing compatibility.

    Algorithm:
    1. For each phenotype stitched cell, keeps the SBS match with minimum distance
    2. For each SBS stitched cell, keeps the phenotype match with minimum distance
    3. Results in 1:1 mapping between stitched cell IDs
    4. Preserves original cell IDs in cell_0/cell_1 columns

    Args:
        raw_matches: DataFrame containing raw cell matches with required columns:
            - stitched_cell_id_0: Phenotype stitched cell identifiers
            - stitched_cell_id_1: SBS stitched cell identifiers
            - cell_0: Original phenotype cell identifiers
            - cell_1: Original SBS cell identifiers
            - distance: Euclidean distance between matched cells

    Returns:
        DataFrame with deduplicated matches, maintaining all input columns
        but with only the best match per stitched cell ID

    Raises:
        ValueError: If required columns are missing from input DataFrame

    Example:
        >>> raw = pd.DataFrame({
        ...     'stitched_cell_id_0': [1, 1, 2],
        ...     'stitched_cell_id_1': [101, 102, 101],
        ...     'cell_0': [1, 1, 2],
        ...     'cell_1': [101, 102, 101],
        ...     'distance': [1.5, 2.0, 1.0]
        ... })
        >>> dedup = legacy_deduplication_stitched_ids(raw)
        >>> len(dedup)  # Should be 2: best match for each stitched ID
        2
    """
    required_cols = [
        "stitched_cell_id_0",
        "stitched_cell_id_1",
        "cell_0",
        "cell_1",
        "distance",
    ]
    missing_cols = [col for col in required_cols if col not in raw_matches.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if raw_matches.empty:
        return raw_matches.copy()

    # Step 1: For each phenotype stitched cell, keep best SBS match
    df_pheno_deduped = raw_matches.sort_values(
        "distance", ascending=True
    ).drop_duplicates("stitched_cell_id_0", keep="first")

    # Step 2: For each SBS stitched cell, keep best phenotype match
    df_final = df_pheno_deduped.sort_values("distance", ascending=True).drop_duplicates(
        "stitched_cell_id_1", keep="first"
    )

    return df_final
