import pandas as pd
import numpy as np
from tifffile import imwrite

from lib.aggregate.montage_utils import create_cell_montage

# read cell data
cell_data = pd.read_hdf(snakemake.input[0])

# subset data to inly include target gene and sgrna combination
cell_data = cell_data[
    (cell_data["gene_symbol_0"] == snakemake.wildcards.gene)
    & (cell_data["sgRNA_0"] == snakemake.wildcards.sgrna)
]

# if this gene/sgrna combination has no cell data, save an empty image
if cell_data.empty:
    imwrite(snakemake.output[0], np.zeros((1, 1), dtype=np.uint8))
else:
    # TODO: Remove when done with testing on Denali data
    # redirects actual paths to paths of Denali data
    def update_file_path(row):
        original_path = row["image_path"]
        parts = original_path.split("/")[-1].split("__")[0].split("_")
        well, tile = parts[0][1:], parts[1][1:]
        new_path = f"/archive/cheeseman/OpticalPooledScreens/lab/barcheese01/screens/denali/input_ph_tif/20X_{well}_Tile-{tile}.phenotype.tif"
        return new_path

    cell_data["image_path"] = cell_data.apply(update_file_path, axis=1)

    # create cell montage
    montage = create_cell_montage(cell_data, snakemake.params.channels)

    # save montage
    imwrite(snakemake.output[0], montage[snakemake.wildcards.channel])
