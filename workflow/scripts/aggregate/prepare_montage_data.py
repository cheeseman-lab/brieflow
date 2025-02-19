from pathlib import Path

import pandas as pd

from lib.aggregate.montage_utils import add_filenames

# load cell data
cell_data = pd.concat([pd.read_parquet(p) for p in snakemake.input], ignore_index=True)

# prepare for montage
prepared_cell_data = add_filenames(
    cell_data, Path(snakemake.params[0]), montage_subset=True
)

# save prepared data
prepared_cell_data.to_parquet(snakemake.output[0])
