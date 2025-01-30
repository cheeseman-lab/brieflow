from pathlib import Path

import pandas as pd

from lib.aggregate.load_format_data import add_filenames

# load cell data
cell_data = pd.read_hdf(snakemake.input[0])

# prepare for montage
prepared_cell_data = add_filenames(
    cell_data, Path(snakemake.params[0]), montage_subset=True
)

# save prepared data
prepared_cell_data.to_hdf(snakemake.output[0], key="x", mode="w")
