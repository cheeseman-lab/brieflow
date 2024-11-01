from lib.sbs_process.compute_standard_deviation import compute_standard_deviation
from skimage.io import imread, imsave

log_filtered_data = imread(snakemake.input[0])

# compute standard deviation
standard_deviation = compute_standard_deviation(
    log_filtered_data=log_filtered_data,
    remove_index=snakemake.params.remove_index,
)

# Save the aligned data as a .tiff file
imsave(snakemake.output[0], standard_deviation)
