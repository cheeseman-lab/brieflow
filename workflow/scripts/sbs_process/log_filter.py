from tifffile import imread, imwrite

from lib.shared.log_filter import log_filter

# load aligned image data
aligned_image_data = imread(snakemake.input[0])

# apply log filter
log_filtered = log_filter(
    aligned_image_data=aligned_image_data,
    skip_index=snakemake.params.skip_index,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], log_filtered)
