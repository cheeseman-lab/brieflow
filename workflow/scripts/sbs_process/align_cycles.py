from lib.sbs_process.align_cycles import align_cycles
from skimage.io import imread, imsave

data = [imread(file_path) for file_path in snakemake.input]

# align cycles
aligned_data = align_cycles(
    data,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    # display_ranges=snakemake.params.display_ranges,
)

# Save the aligned data as a .tiff file
imsave(snakemake.output[0], aligned_data)
