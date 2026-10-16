"""Add one-row-per-nucleus MN summaries without multiplying phenotype rows."""

import pandas as pd

keys = ["plate", "well", "tile", "label"]
dtypes = {key: str for key in keys[:-1]}
phenotype = pd.read_csv(snakemake.input.phenotype, sep="\t", dtype=dtypes)
summary = pd.read_csv(snakemake.input.summary, sep="\t", dtype=dtypes)
if any(column in phenotype for column in summary if column not in keys):
    raise ValueError("MN summary columns already present in phenotype table")
merged = phenotype.merge(summary, on=keys, how="left", validate="one_to_one")
if merged.mn_domain_available.isna().any():
    raise ValueError("Some phenotype nuclei have no corresponding MN parent record")
merged.to_csv(snakemake.output[0], sep="\t", index=False)
