from lib.shared.io import read_image, save_image
from lib.shared.log_filter import log_filter

# load aligned image data
aligned_image_data = read_image(snakemake.input[0])

# apply log filter
log_filtered = log_filter(
    aligned_image_data=aligned_image_data,
    skip_index=snakemake.params.skip_index,
)

# Save the log filtered data
save_image(
    log_filtered,
    snakemake.output[0],
    pixel_size_z=snakemake.params.pixel_size_z,
    pixel_size_y=snakemake.params.pixel_size_y,
    pixel_size_x=snakemake.params.pixel_size_x,
    channel_names=snakemake.params.channel_names,
)
