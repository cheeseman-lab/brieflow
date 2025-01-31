import pandas as pd

from lib.aggregate.cell_classification import split_mitotic_simple

# load standardized data
standardized_data = pd.read_hdf(snakemake.input[0])

# split mitotic and simple cells
mitotic_data, interphase_data = split_mitotic_simple(
    standardized_data, snakemake.params["threshold_conditions"]
)

# save split data
mitotic_data.to_hdf(snakemake.output[0], key="x", mode="w")
interphase_data.to_hdf(snakemake.output[1], key="x", mode="w")
