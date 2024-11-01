from lib.sbs_processing.align_cycles import align_cycles
from skimage.io import imsave

# align cycles
aligned_data = align_cycles(
    snakemake.input,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    display_ranges=snakemake.params.display_ranges,
)

# Save the aligned data as a .tiff file
imsave(snakemake.output[0], aligned_data)
