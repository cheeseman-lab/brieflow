"""
Well Deduplication Library - Production version for legacy-compatible spatial deduplication.
Focuses on stitched cell ID deduplication with comprehensive validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def validate_final_matches(final_matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate final matches for spatial accuracy and quality.
    
    Args:
        final_matches: DataFrame with final cell matches
        
    Returns:
        Dictionary with comprehensive validation results
    """
    if final_matches.empty:
        return {
            'status': 'empty',
            'match_count': 0,
            'is_1to1_stitched': True,  # Technically true for empty set
            'validation_status': 'empty'
        }
    
    # Check for stitched ID duplicates (should be 1:1 for spatial accuracy)
    has_stitched_ids = ('stitched_cell_id_0' in final_matches.columns and 
                       'stitched_cell_id_1' in final_matches.columns)
    
    if has_stitched_ids:
        stitched_pheno_dups = final_matches['stitched_cell_id_0'].duplicated().sum()
        stitched_sbs_dups = final_matches['stitched_cell_id_1'].duplicated().sum()
        is_1to1_stitched = stitched_pheno_dups == 0 and stitched_sbs_dups == 0
    else:
        stitched_pheno_dups = 0
        stitched_sbs_dups = 0
        is_1to1_stitched = False
    
    # Check original ID duplicates (may exist - acceptable for legacy compatibility)
    original_pheno_dups = final_matches['cell_0'].duplicated().sum()
    original_sbs_dups = final_matches['cell_1'].duplicated().sum()
    
    # Distance statistics
    distances = final_matches['distance']
    
    # Quality distribution
    under_1px = (distances < 1).sum()
    under_2px = (distances < 2).sum()
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_20px = (distances > 20).sum()
    over_50px = (distances > 50).sum()
    
    # Overall quality assessment
    precision_5px = under_5px / len(distances)
    mean_dist = distances.mean()
    
    if mean_dist < 2.0 and precision_5px > 0.8 and over_50px == 0:
        quality_tier = 'excellent'
    elif mean_dist < 5.0 and precision_5px > 0.6 and over_50px == 0:
        quality_tier = 'good'
    elif over_50px == 0:
        quality_tier = 'acceptable'
    else:
        quality_tier = 'poor'
    
    return {
        'status': 'valid',
        'match_count': len(final_matches),
        'is_1to1_stitched': is_1to1_stitched,
        'has_stitched_ids': has_stitched_ids,
        'stitched_duplicates': {
            'phenotype': int(stitched_pheno_dups),
            'sbs': int(stitched_sbs_dups)
        },
        'original_duplicates': {
            'phenotype': int(original_pheno_dups),
            'sbs': int(original_sbs_dups)
        },
        'distance_stats': {
            'mean': float(distances.mean()),
            'median': float(distances.median()),
            'std': float(distances.std()),
            'min': float(distances.min()),
            'max': float(distances.max()),
            'p95': float(distances.quantile(0.95))
        },
        'distance_distribution': {
            'under_1px': int(under_1px),
            'under_2px': int(under_2px),
            'under_5px': int(under_5px),
            'under_10px': int(under_10px),
            'over_20px': int(over_20px),
            'over_50px': int(over_50px)
        },
        'quality_metrics': {
            'precision_1px': float(under_1px / len(distances)),
            'precision_2px': float(under_2px / len(distances)),
            'precision_5px': float(precision_5px),
            'precision_10px': float(under_10px / len(distances)),
            'quality_tier': quality_tier
        },
        'validation_status': 'valid' if is_1to1_stitched else 'stitched_duplicates_present',
        'issues_detected': {
            'large_distances': over_50px > 0,
            'stitched_duplicates': not is_1to1_stitched and has_stitched_ids,
            'no_stitched_ids': not has_stitched_ids
        }
    }


def analyze_duplicates(raw_matches: pd.DataFrame, id_column_0: str = 'cell_0', id_column_1: str = 'cell_1') -> Dict[str, Any]:
    """
    Analyze duplication patterns in raw matches for any ID columns.
    
    Args:
        raw_matches: DataFrame with raw cell matches
        id_column_0: Column name for phenotype cell IDs (default: 'cell_0')
        id_column_1: Column name for SBS cell IDs (default: 'cell_1')
        
    Returns:
        Dictionary with duplication analysis
    """
    if raw_matches.empty or id_column_0 not in raw_matches.columns or id_column_1 not in raw_matches.columns:
        return {
            'unique_phenotype_cells': 0,
            'unique_sbs_cells': 0,
            'multi_match_phenotype': 0,
            'multi_match_sbs': 0,
            'max_phenotype_matches': 0,
            'max_sbs_matches': 0,
            'duplication_rate_phenotype': 0.0,
            'duplication_rate_sbs': 0.0
        }
    
    # Count matches per cell
    pheno_counts = raw_matches[id_column_0].value_counts()
    sbs_counts = raw_matches[id_column_1].value_counts()
    
    # Analysis
    unique_phenotype = len(pheno_counts)
    unique_sbs = len(sbs_counts)
    
    multi_match_phenotype = (pheno_counts > 1).sum()
    multi_match_sbs = (sbs_counts > 1).sum()
    
    max_phenotype_matches = pheno_counts.max() if len(pheno_counts) > 0 else 0
    max_sbs_matches = sbs_counts.max() if len(sbs_counts) > 0 else 0
    
    return {
        'id_columns_analyzed': f"{id_column_0}, {id_column_1}",
        'unique_phenotype_cells': int(unique_phenotype),
        'unique_sbs_cells': int(unique_sbs),
        'multi_match_phenotype': int(multi_match_phenotype),
        'multi_match_sbs': int(multi_match_sbs),
        'max_phenotype_matches': int(max_phenotype_matches),
        'max_sbs_matches': int(max_sbs_matches),
        'duplication_rate_phenotype': float(multi_match_phenotype / unique_phenotype) if unique_phenotype > 0 else 0.0,
        'duplication_rate_sbs': float(multi_match_sbs / unique_sbs) if unique_sbs > 0 else 0.0
    }


def legacy_deduplication_stitched_ids(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Apply legacy-compatible deduplication using stitched cell IDs for spatial accuracy.
    
    This is the core deduplication function that:
    1. For each phenotype stitched cell, keeps the best SBS match
    2. For each SBS stitched cell, keeps the best phenotype match
    3. Preserves original cell IDs in cell_0/cell_1 for downstream compatibility
    
    Args:
        raw_matches: DataFrame with raw cell matches including stitched_cell_id columns
        
    Returns:
        DataFrame with deduplicated matches
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['stitched_cell_id_0', 'stitched_cell_id_1', 'cell_0', 'cell_1', 'distance']
    missing_cols = [col for col in required_cols if col not in raw_matches.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if raw_matches.empty:
        return raw_matches.copy()
    
    # Step 1: For each phenotype stitched cell, keep best SBS match
    df_pheno_deduped = (raw_matches
                       .sort_values('distance', ascending=True)
                       .drop_duplicates('stitched_cell_id_0', keep='first'))
    
    # Step 2: For each SBS stitched cell, keep best phenotype match
    df_final = (df_pheno_deduped
               .sort_values('distance', ascending=True)
               .drop_duplicates('stitched_cell_id_1', keep='first'))
    
    return df_final


def get_quality_summary(matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a concise quality summary for reporting.
    
    Args:
        matches: DataFrame with final matches
        
    Returns:
        Dictionary with key quality metrics for logging/reporting
    """
    if matches.empty:
        return {
            'match_count': 0,
            'mean_distance': 0.0,
            'quality_tier': 'empty',
            'precision_5px': 0.0
        }
    
    distances = matches['distance']
    under_5px = (distances < 5).sum()
    over_50px = (distances > 50).sum()
    
    precision_5px = under_5px / len(distances)
    mean_dist = distances.mean()
    
    # Determine quality tier
    if mean_dist < 2.0 and precision_5px > 0.8 and over_50px == 0:
        quality_tier = 'excellent'
    elif mean_dist < 5.0 and precision_5px > 0.6 and over_50px == 0:
        quality_tier = 'good'
    elif over_50px == 0:
        quality_tier = 'acceptable'
    else:
        quality_tier = 'poor'
    
    return {
        'match_count': len(matches),
        'mean_distance': float(mean_dist),
        'median_distance': float(distances.median()),
        'max_distance': float(distances.max()),
        'precision_5px': float(precision_5px),
        'precision_10px': float((distances < 10).sum() / len(distances)),
        'large_distance_count': int(over_50px),
        'quality_tier': quality_tier
    }