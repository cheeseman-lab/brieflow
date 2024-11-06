from lib.sbs_process.segment_cellpose import segment_cellpose
from skimage.io import imread, imsave

# load illumination corrected data
illumination_corrected_data = imread(snakemake.input[0])

# segment cells using cellpose
segmented_data = segment_cellpose(
    data=illumination_corrected_data,
    dapi_index=snakemake.params.dapi_index,
    cyto_index=snakemake.params.cyto_index,
    nuclei_diameter=snakemake.params.nuclei_diameter,
    cell_diameter=snakemake.params.cell_diameter,
)

# save segmented data
imsave(snakemake.output[0], segmented_data)
