import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.merge.deduplicate_merge import deduplicate_cells, check_matching_rates, analyze_distance_distribution
from lib.merge.format_merge import identify_single_gene_mappings

# Load data for evaluating merge (same as before)
merge_formatted = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_cells = validate_dtypes(pd.read_parquet(snakemake.input[1]))
phenotype_min_cp = validate_dtypes(pd.read_parquet(snakemake.input[2]))

# Get approach from params
approach = getattr(snakemake.params, 'approach', 'fast')

# Deduplicate cells using unified function
merge_deduplicated, deduplication_stats = deduplicate_cells(
    merge_formatted, 
    mapped_single_gene=False, 
    return_stats=True,
    approach=approach  # Pass approach parameter
)

# Add distance distribution analysis (new functionality)
distance_analysis = analyze_distance_distribution(merge_deduplicated)

# Add distance analysis to deduplication stats if we have distance data
if distance_analysis:
    print(f"\nDistance distribution analysis:")
    print(f"Mean distance: {distance_analysis['distance_stats']['mean']:.2f}px")
    print(f"<5px precision: {distance_analysis['distance_distribution']['under_5px']}/{len(merge_deduplicated)} ({distance_analysis['distance_distribution']['under_5px']/len(merge_deduplicated)*100:.1f}%)")

# Save deduplication stats (same as before)
deduplication_stats.to_csv(snakemake.output[0], sep="\t", index=False)
merge_deduplicated.to_parquet(snakemake.output[1])

# Rest of the script stays exactly the same...
sbs_cells["mapped_single_gene"] = sbs_cells.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)

sbs_rates = check_matching_rates(
    sbs_cells, merge_deduplicated, modality="sbs", return_stats=True
)
sbs_rates.to_csv(snakemake.output[2], sep="\t", index=False)

phenotype_rates = check_matching_rates(
    phenotype_min_cp, merge_deduplicated, modality="phenotype", return_stats=True
)
phenotype_rates.to_csv(snakemake.output[3], sep="\t", index=False)