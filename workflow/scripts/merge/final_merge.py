from lib.merge.final_merge import final_merge

# Thin wrapper over the shared streaming merge so every caller uses identical
# logic (see lib/merge/final_merge.py).
final_merge(
    deduplicated_path=snakemake.input[0],
    phenotype_cp_path=snakemake.input[1],
    output_path=snakemake.output[0],
    approach=snakemake.params.approach,
    exclude_markers=snakemake.params.exclude_markers,
)
