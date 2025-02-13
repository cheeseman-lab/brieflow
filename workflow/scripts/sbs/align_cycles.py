from tifffile import imread, imwrite

from lib.sbs.align_cycles import align_cycles

# load image data
image_data = [imread(file_path) for file_path in snakemake.input]

# align cycles
aligned_data = align_cycles(
    image_data,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    keep_extras=snakemake.params.keep_extras,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
