import pandas as pd
import numpy as np
from tifffile import imwrite

from lib.aggregate.montage_utils import create_cell_montage

# read cell data
montage_data = pd.read_csv(snakemake.input[0], sep="\t")

# create cell montage
montage = create_cell_montage(montage_data, snakemake.params.channels)

# save montages
for index, channel_montage in enumerate(montage.values()):
    imwrite(snakemake.output[index], channel_montage)
