from skimage.io import imsave

from lib.shared.file_utils import read_stack
from lib.phenotype_process.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)

# load nuclei and cell segmentation data
nuclei = read_stack(snakemake.input[0])
cells = read_stack(snakemake.input[1])

# identify cytoplasms with cellpose
cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)

# save cytoplasms data
imsave(snakemake.output[0], cytoplasms)
