"""
Step 3 Library: Advanced deduplication functions.
Save this as: lib/merge/deduplication.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.optimize import linear_sum_assignment


def analyze_duplicates(raw_matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze duplication patterns in raw matches.
    
    Args:
        raw_matches: DataFrame with raw cell matches
        
    Returns:
        Dictionary with duplication analysis
    """
    if raw_matches.empty:
        return {
            'unique_phenotype_cells': 0,
            'unique_sbs_cells': 0,
            'multi_match_phenotype': 0,
            'multi_match_sbs': 0,
            'max_phenotype_matches': 0,
            'max_sbs_matches': 0
        }
    
    # Count matches per cell
    pheno_counts = raw_matches['cell_0'].value_counts()
    sbs_counts = raw_matches['cell_1'].value_counts()
    
    # Analysis
    unique_phenotype = len(pheno_counts)
    unique_sbs = len(sbs_counts)
    
    multi_match_phenotype = (pheno_counts > 1).sum()
    multi_match_sbs = (sbs_counts > 1).sum()
    
    max_phenotype_matches = pheno_counts.max() if len(pheno_counts) > 0 else 0
    max_sbs_matches = sbs_counts.max() if len(sbs_counts) > 0 else 0
    
    return {
        'unique_phenotype_cells': int(unique_phenotype),
        'unique_sbs_cells': int(unique_sbs),
        'multi_match_phenotype': int(multi_match_phenotype),
        'multi_match_sbs': int(multi_match_sbs),
        'max_phenotype_matches': int(max_phenotype_matches),
        'max_sbs_matches': int(max_sbs_matches),
        'duplication_rate_phenotype': float(multi_match_phenotype / unique_phenotype) if unique_phenotype > 0 else 0.0,
        'duplication_rate_sbs': float(multi_match_sbs / unique_sbs) if unique_sbs > 0 else 0.0
    }


def greedy_1to1_matching(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy 1:1 matching - assigns best matches first, ensuring no cell is used twice.
    
    Args:
        raw_matches: DataFrame with raw cell matches
        
    Returns:
        DataFrame with 1:1 matches
    """
    if raw_matches.empty:
        return raw_matches.copy()
    
    print(f"Applying greedy 1:1 matching to {len(raw_matches):,} raw matches")
    
    # Sort by distance (best matches first)
    sorted_matches = raw_matches.sort_values('distance').reset_index(drop=True)
    
    # Keep track of used cells
    used_phenotype = set()
    used_sbs = set()
    final_matches = []
    
    for _, match in sorted_matches.iterrows():
        pheno_cell = match['cell_0']
        sbs_cell = match['cell_1']
        
        # Skip if either cell is already used
        if pheno_cell in used_phenotype or sbs_cell in used_sbs:
            continue
        
        # Add this match
        final_matches.append(match)
        
        # Mark cells as used
        used_phenotype.add(pheno_cell)
        used_sbs.add(sbs_cell)
    
    if final_matches:
        result = pd.DataFrame(final_matches).reset_index(drop=True)
        print(f"Greedy 1:1 matching: {len(result):,} final matches")
        return result
    else:
        return pd.DataFrame(columns=raw_matches.columns)


def hungarian_1to1_matching(raw_matches: pd.DataFrame, max_cells: int = 10000) -> pd.DataFrame:
    """
    Hungarian algorithm for optimal 1:1 matching (for smaller datasets).
    
    Args:
        raw_matches: DataFrame with raw cell matches
        max_cells: Maximum number of cells to use Hungarian on (for performance)
        
    Returns:
        DataFrame with 1:1 matches
    """
    if raw_matches.empty:
        return raw_matches.copy()
    
    print(f"Applying Hungarian 1:1 matching to {len(raw_matches):,} raw matches")
    
    # Get unique cells
    unique_phenotype = raw_matches['cell_0'].unique()
    unique_sbs = raw_matches['cell_1'].unique()
    
    # If too many cells, fall back to greedy
    if len(unique_phenotype) > max_cells or len(unique_sbs) > max_cells:
        print(f"Too many cells for Hungarian ({len(unique_phenotype)}, {len(unique_sbs)}), using greedy instead")
        return greedy_1to1_matching(raw_matches)
    
    # Create cell index mappings
    pheno_to_idx = {cell: idx for idx, cell in enumerate(unique_phenotype)}
    sbs_to_idx = {cell: idx for idx, cell in enumerate(unique_sbs)}
    
    # Create cost matrix (use distance as cost)
    n_pheno = len(unique_phenotype)
    n_sbs = len(unique_sbs)
    
    # Initialize with high cost (no match)
    cost_matrix = np.full((n_pheno, n_sbs), 1000.0)
    
    # Fill in actual distances
    for _, match in raw_matches.iterrows():
        pheno_idx = pheno_to_idx[match['cell_0']]
        sbs_idx = sbs_to_idx[match['cell_1']]
        cost_matrix[pheno_idx, sbs_idx] = match['distance']
    
    # Apply Hungarian algorithm
    pheno_indices, sbs_indices = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches (distance < threshold)
    final_matches = []
    for p_idx, s_idx in zip(pheno_indices, sbs_indices):
        cost = cost_matrix[p_idx, s_idx]
        if cost < 1000.0:  # Valid match
            pheno_cell = unique_phenotype[p_idx]
            sbs_cell = unique_sbs[s_idx]
            
            # Find the original match data
            match_data = raw_matches[
                (raw_matches['cell_0'] == pheno_cell) & 
                (raw_matches['cell_1'] == sbs_cell)
            ].iloc[0]
            
            final_matches.append(match_data)
    
    if final_matches:
        result = pd.DataFrame(final_matches).reset_index(drop=True)
        print(f"Hungarian 1:1 matching: {len(result):,} final matches")
        return result
    else:
        return pd.DataFrame(columns=raw_matches.columns)


def validate_final_matches(final_matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate final matches for 1:1 property and quality.
    
    Args:
        final_matches: DataFrame with final cell matches
        
    Returns:
        Dictionary with validation results
    """
    if final_matches.empty:
        return {
            'status': 'empty',
            'is_1to1': True,  # Technically true for empty set
            'match_count': 0
        }
    
    # Check for duplicates
    pheno_duplicates = final_matches['cell_0'].duplicated().sum()
    sbs_duplicates = final_matches['cell_1'].duplicated().sum()
    
    is_1to1 = pheno_duplicates == 0 and sbs_duplicates == 0
    
    # Distance statistics
    distances = final_matches['distance']
    
    # Quality metrics
    under_1px = (distances < 1).sum()
    under_2px = (distances < 2).sum()
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_20px = (distances > 20).sum()
    over_50px = (distances > 50).sum()
    
    # Overall quality assessment
    excellent_quality = (
        is_1to1 and 
        distances.mean() < 2.0 and 
        over_20px == 0 and
        under_5px / len(distances) > 0.8
    )
    
    good_quality = (
        is_1to1 and 
        distances.mean() < 5.0 and 
        over_50px == 0 and
        under_10px / len(distances) > 0.7
    )
    
    return {
        'status': 'valid',
        'match_count': len(final_matches),
        'is_1to1': is_1to1,
        'duplicates': {
            'phenotype': int(pheno_duplicates),
            'sbs': int(sbs_duplicates)
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
        'quality_assessment': {
            'excellent': excellent_quality,
            'good': good_quality,
            'acceptable': is_1to1,
            'issues': not is_1to1 or over_50px > 0
        },
        'efficiency_metrics': {
            'precision_1px': float(under_1px / len(distances)),
            'precision_2px': float(under_2px / len(distances)),
            'precision_5px': float(under_5px / len(distances)),
            'precision_10px': float(under_10px / len(distances))
        }
    }


def compare_deduplication_methods(raw_matches: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare different deduplication methods on the same dataset.
    
    Args:
        raw_matches: DataFrame with raw cell matches
        
    Returns:
        Dictionary comparing methods
    """
    print(f"Comparing deduplication methods on {len(raw_matches):,} raw matches")
    
    results = {}
    
    # Simple deduplication (keep best match per phenotype cell)
    simple_matches = raw_matches.sort_values('distance').drop_duplicates('cell_0', keep='first')
    results['simple'] = {
        'final_matches': len(simple_matches),
        'mean_distance': float(simple_matches['distance'].mean()) if len(simple_matches) > 0 else 0.0,
        'validation': validate_final_matches(simple_matches)
    }
    
    # Greedy 1:1 matching
    greedy_matches = greedy_1to1_matching(raw_matches)
    results['greedy_1to1'] = {
        'final_matches': len(greedy_matches),
        'mean_distance': float(greedy_matches['distance'].mean()) if len(greedy_matches) > 0 else 0.0,
        'validation': validate_final_matches(greedy_matches)
    }
    
    # Hungarian 1:1 matching (if feasible)
    unique_phenotype = raw_matches['cell_0'].nunique()
    unique_sbs = raw_matches['cell_1'].nunique()
    
    if unique_phenotype <= 5000 and unique_sbs <= 5000:
        hungarian_matches = hungarian_1to1_matching(raw_matches)
        results['hungarian_1to1'] = {
            'final_matches': len(hungarian_matches),
            'mean_distance': float(hungarian_matches['distance'].mean()) if len(hungarian_matches) > 0 else 0.0,
            'validation': validate_final_matches(hungarian_matches)
        }
    else:
        results['hungarian_1to1'] = {
            'status': 'skipped',
            'reason': 'too_many_cells',
            'unique_phenotype': unique_phenotype,
            'unique_sbs': unique_sbs
        }
    
    # Summary comparison
    method_scores = {}
    for method, data in results.items():
        if 'validation' in data:
            validation = data['validation']
            # Score based on: 1:1 property, match count, and distance quality
            score = 0
            if validation['is_1to1']:
                score += 40  # 1:1 is critical
            score += min(data['final_matches'] / 1000, 30)  # More matches is better (up to 30 points)
            if data['mean_distance'] < 5.0:
                score += 20  # Good distance quality
            elif data['mean_distance'] < 10.0:
                score += 10
            
            method_scores[method] = score
    
    # Find best method
    best_method = max(method_scores.items(), key=lambda x: x[1])[0] if method_scores else 'simple'
    
    results['comparison'] = {
        'best_method': best_method,
        'method_scores': method_scores,
        'raw_matches_input': len(raw_matches)
    }
    
    return results