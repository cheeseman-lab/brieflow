from lib.sbs.compute_standard_deviation import compute_standard_deviation
from lib.shared.io import read_image, save_image

# load log filtered image data (supports TIFF and Zarr)
log_filtered_data = read_image(snakemake.input[0])

# compute standard deviation
standard_deviation = compute_standard_deviation(
    log_filtered_data=log_filtered_data,
    remove_index=snakemake.params.remove_index,
)

# Save the standard deviation data
save_image(standard_deviation, snakemake.output[0])
