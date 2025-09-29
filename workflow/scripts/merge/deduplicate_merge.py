"""Snakemake script for cell deduplication in optical pooled screening workflows.

This script performs two-step deduplication of merged phenotype-SBS cell mappings,
evaluates cell retention rates, and generates quality control metrics for the
merge process. Supports both tile-based and stitched image approaches.
"""

import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.merge.deduplicate_merge import (
    deduplicate_cells,
    check_matching_rates,
    analyze_distance_distribution,
)

# Load input datasets with data type validation
merge_formatted = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_cells = validate_dtypes(pd.read_parquet(snakemake.input[1]))
phenotype_min_cp = validate_dtypes(pd.read_parquet(snakemake.input[2]))

# Extract configuration parameters
approach = getattr(snakemake.params, "approach", "fast")
dedup_step1 = getattr(snakemake.params, "dedup_step1", None)
dedup_step2 = getattr(snakemake.params, "dedup_step2", None)

# Perform two-step cell deduplication with statistics tracking
merge_deduplicated, deduplication_stats = deduplicate_cells(
    merge_formatted,
    mapped_single_gene=False,
    return_stats=True,
    approach=approach,
    sbs_dedup_prior=dedup_step1,
    pheno_dedup_prior=dedup_step2,
)

# Analyze spatial alignment quality through distance distribution
distance_analysis = analyze_distance_distribution(merge_deduplicated)

# Display distance distribution summary if analysis succeeded
if distance_analysis:
    dist_stats = distance_analysis['distance_stats']
    dist_counts = distance_analysis['distance_distribution']
    
    print(f"\nSpatial alignment quality metrics:")
    print(f"Mean distance: {dist_stats['mean']:.2f}px")
    print(f"Median distance: {dist_stats['median']:.2f}px")
    print(f"High-precision alignments (<5px): {dist_counts['under_5px']}/{len(merge_deduplicated)} "
          f"({dist_counts['under_5px'] / len(merge_deduplicated) * 100:.1f}%)")

# Export deduplication statistics
deduplication_stats.to_csv(snakemake.output[0], sep="\t", index=False)
merge_deduplicated.to_parquet(snakemake.output[1])

# Calculate and export SBS matching statistics
sbs_rates = check_matching_rates(
    sbs_cells, merge_deduplicated, modality="sbs", return_stats=True
)
sbs_rates.to_csv(snakemake.output[2], sep="\t", index=False)

# Calculate and export phenotype matching statistics
phenotype_rates = check_matching_rates(
    phenotype_min_cp, merge_deduplicated, modality="phenotype", return_stats=True
)
phenotype_rates.to_csv(snakemake.output[3], sep="\t", index=False)