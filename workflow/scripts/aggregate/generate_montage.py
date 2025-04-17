import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.io import imsave

from lib.aggregate.montage_utils import create_cell_montage

# read cell data
montage_data = pd.read_csv(snakemake.input[0], sep="\t")

# create cell montage
montage = create_cell_montage(montage_data, snakemake.params.channels)

# save montages
for index, channel_montage in enumerate(montage.values()):
    # Normalize to 0–255 for PNG
    montage_uint8 = rescale_intensity(
        channel_montage, in_range="image", out_range=(0, 255)
    ).astype(np.uint8)

    imsave(snakemake.output[index], montage_uint8)
