"""Stitch Merge - Cell-to-cell matching using alignment parameters.

Performs cell-to-cell matching between phenotype and SBS datasets within a well
using spatial alignment transformations.
"""

import pandas as pd
from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_merge import (
    load_alignment_parameters,
    find_cell_matches,
    filter_tiles_by_diversity,
    build_final_matches,
    create_merge_summary,
)

# Load all inputs
phenotype_scaled = validate_dtypes(
    pd.read_parquet(snakemake.input.scaled_phenotype_positions)
)
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))
alignment_params = validate_dtypes(pd.read_parquet(snakemake.input.alignment_params))
phenotype_transformed = validate_dtypes(
    pd.read_parquet(snakemake.input.transformed_phenotype_positions)
)

plate = snakemake.params.plate
well = snakemake.params.well
threshold = snakemake.params.threshold

print(f"Processing Plate {plate}, Well {well}")
print(
    f"Input: {len(phenotype_scaled):,} phenotype cells, {len(sbs_positions):,} SBS cells"
)
print(f"Distance threshold: {threshold} px")

# Filter tiles by diversity
phenotype_filtered = filter_tiles_by_diversity(phenotype_scaled, "Phenotype")
sbs_filtered = filter_tiles_by_diversity(sbs_positions, "SBS")

# Filter transformed positions to match
phenotype_transformed_filtered = phenotype_transformed[
    phenotype_transformed.index.isin(phenotype_filtered.index)
]

print(
    f"After filtering: {len(phenotype_filtered):,} phenotype, {len(sbs_filtered):,} SBS cells"
)

# Load alignment and find matches
alignment = load_alignment_parameters(alignment_params.iloc[0])
print(
    f"Using {alignment.get('approach', 'unknown')} alignment (score: {alignment.get('score', 0):.3f})"
)

raw_matches, match_stats = find_cell_matches(
    phenotype_positions=phenotype_filtered,
    sbs_positions=sbs_filtered,
    alignment=alignment,
    threshold=threshold,
    transformed_phenotype_positions=phenotype_transformed_filtered,
)

# Build final output with metadata
final_matches = build_final_matches(
    raw_matches=raw_matches,
    phenotype_filtered=phenotype_filtered,
    sbs_filtered=sbs_filtered,
    plate=plate,
    well=well,
)

# Save outputs
final_matches.to_parquet(str(snakemake.output.raw_matches))
final_matches.to_parquet(str(snakemake.output.merged_cells))

# Create summary
summary_df = create_merge_summary(
    final_matches=final_matches,
    phenotype_scaled=phenotype_scaled,
    sbs_positions=sbs_positions,
    phenotype_filtered=phenotype_filtered,
    sbs_filtered=sbs_filtered,
    alignment=alignment,
    threshold=threshold,
    plate=plate,
    well=well,
)

summary_df.to_csv(str(snakemake.output.merge_summary), sep="\t", index=False)

print(f"Completed: {len(final_matches):,} matched cells")
