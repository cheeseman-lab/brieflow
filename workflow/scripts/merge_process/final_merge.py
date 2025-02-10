import pandas as pd

# Load deduplicated merge data
merge_deduplicated = pd.read_hdf(snakemake.input[1])

# Load full feature data
cp_phenotype = pd.read_hdf(snakemake.input[4])

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["well", "tile", "cell_0"],
)

# Save final merged dataset
merged_final.to_hdf(snakemake.output[0], "x", mode="w")
