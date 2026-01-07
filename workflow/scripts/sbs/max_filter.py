from lib.shared.io import read_image, save_image
from lib.sbs.max_filter import max_filter

# load log filtered image data
log_filtered_data = read_image(snakemake.input[0])

# apply max filter
max_filtered = max_filter(
    log_filtered_data=log_filtered_data,
    width=snakemake.params.width,
    remove_index=snakemake.params.remove_index,
)

# Save the max filtered data
save_image(
    max_filtered,
    snakemake.output[0],
    pixel_size_z=snakemake.params.pixel_size_z,
    pixel_size_y=snakemake.params.pixel_size_y,
    pixel_size_x=snakemake.params.pixel_size_x,
    channel_names=snakemake.params.channel_names,
)
