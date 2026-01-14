from tifffile import imwrite

from lib.sbs.max_filter import max_filter
from lib.shared.io import read_image

# load log filtered image data (supports TIFF and Zarr)
log_filtered_data = read_image(snakemake.input[0])

# apply max filter
max_filtered = max_filter(
    log_filtered_data=log_filtered_data,
    width=snakemake.params.width,
    remove_index=snakemake.params.remove_index,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], max_filtered)
