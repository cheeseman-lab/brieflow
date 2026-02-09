import numpy as np
from tifffile import imread, imwrite

from lib.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)

# load nuclei and cell segmentation data
nuclei = imread(snakemake.input[0])
cells = imread(snakemake.input[1])

# check if cell segmentation is enabled
segment_cells = snakemake.params.get("segment_cells", True)

if segment_cells:
    # identify cytoplasms with cellpose
    cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)
else:
    # write blank array when cell segmentation is disabled
    cytoplasms = np.zeros_like(nuclei, dtype=np.int32)

# save cytoplasms data
imwrite(snakemake.output[0], cytoplasms)
