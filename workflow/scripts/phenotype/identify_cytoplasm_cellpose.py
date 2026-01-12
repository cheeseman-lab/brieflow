from tifffile import imwrite

from lib.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)
from lib.shared.io import read_image

# load nuclei and cell segmentation data (supports TIFF and Zarr)
nuclei = read_image(snakemake.input[0])
cells = read_image(snakemake.input[1])

# identify cytoplasms with cellpose
cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)

# save cytoplasms data
imwrite(snakemake.output[0], cytoplasms)
