import numpy as np

from lib.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)
from lib.shared.io import read_image, save_image

# Load nuclei and cell segmentation data
nuclei = read_image(snakemake.input[0])
cells = read_image(snakemake.input[1])

# Check if cell segmentation is enabled
segment_cells = snakemake.params.get("segment_cells", True)

if segment_cells:
    # identify cytoplasms with cellpose
    cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)
else:
    # write blank array when cell segmentation is disabled
    cytoplasms = np.zeros_like(nuclei, dtype=np.int32)

# Save cytoplasms data
save_image(cytoplasms, snakemake.output[0], is_label=True)
