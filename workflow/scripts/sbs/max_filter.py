from lib.sbs.max_filter import max_filter
from lib.shared.io import read_image, save_image

# Load log filtered image data
log_filtered_data = read_image(snakemake.input[0])

# Apply max filter
max_filtered = max_filter(
    log_filtered_data=log_filtered_data,
    width=snakemake.params.width,
    remove_index=snakemake.params.remove_index,
)

# Save the max filtered data
save_image(max_filtered, snakemake.output[0])
