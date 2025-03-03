import pandas as pd

from lib.merge.deduplicate_merge import deduplicate_cells, check_matching_rates
from lib.merge.format_merge import identify_single_gene_mappings

# Load data for evaluating merge
merge_cleaned = pd.read_parquet(snakemake.input[0])
sbs_cells = pd.read_parquet(snakemake.input[1])
phenotype_min_cp = pd.read_parquet(snakemake.input[2])

# Deduplicate cells and save results
merge_deduplicated, deduplication_stats = deduplicate_cells(
    merge_cleaned, mapped_single_gene=False, return_stats=True
)
deduplication_stats.to_csv(snakemake.output[0], sep="\t", index=False)
merge_deduplicated.to_parquet(snakemake.output[1])

# Identify single gene mappings in SBS
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
