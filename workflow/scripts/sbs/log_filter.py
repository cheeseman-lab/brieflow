from lib.shared.log_filter import log_filter
from lib.shared.io import read_image, save_image

# Load aligned image data
aligned_image_data = read_image(snakemake.input[0])

# Apply log filter
log_filtered = log_filter(
    aligned_image_data=aligned_image_data,
    skip_index=snakemake.params.skip_index,
)

# Save the log filtered data
save_image(log_filtered, snakemake.output[0])
