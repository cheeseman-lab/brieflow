from tifffile import imread, imsave

from lib.sbs_process.max_filter import max_filter

# load log filtered image data
log_filtered_data = imread(snakemake.input[0])

# apply max filter
max_filtered = max_filter(
    log_filtered_data=log_filtered_data,
    width=snakemake.params.width,
    remove_index=snakemake.params.remove_index,
)

# Save the aligned data as a .tiff file
imsave(snakemake.output[0], max_filtered)
