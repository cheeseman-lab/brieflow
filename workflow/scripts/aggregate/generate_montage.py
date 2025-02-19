import pandas as pd
import numpy as np
from tifffile import imwrite

from lib.aggregate.montage_utils import create_cell_montage

# read cell data
cell_data = pd.read_parquet(snakemake.input[0])

# subset data to inly include target gene and sgrna combination
cell_data = cell_data[
    (cell_data["gene_symbol_0"] == snakemake.wildcards.gene)
    & (cell_data["sgRNA_0"] == snakemake.wildcards.sgrna)
]

# if this gene/sgrna combination has no cell data, save an empty image
if cell_data.empty:
    imwrite(snakemake.output[0], np.zeros((1, 1), dtype=np.uint8))
else:
    # create cell montage
    montage = create_cell_montage(cell_data, snakemake.params.channels)

    # save montage
    imwrite(snakemake.output[0], montage[snakemake.wildcards.channel])
