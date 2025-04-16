import pandas as pd

# load aggregated data
aggregated_data = pd.read_csv(snakemake.input[0], sep="\t")

# filter by cell cutoff
min_cell_cutoff = snakemake.params.min_cell_cutoffs[snakemake.params.cell_class]
aggregated_data = aggregated_data[aggregated_data["cell_count"] > min_cell_cutoff]

# save filtered data
aggregated_data.to_csv(snakemake.output[0], sep="\t", index=False)
