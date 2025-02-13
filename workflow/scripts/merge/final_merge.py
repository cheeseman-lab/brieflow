import pandas as pd

# Load deduplicated merge data
merge_deduplicated = pd.read_parquet(snakemake.input[0])

# Load full feature data
cp_phenotype = pd.read_parquet(snakemake.input[1])

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["tile", "cell_0"],
)

# Save final merged dataset
merged_final.to_parquet(snakemake.output[0])
