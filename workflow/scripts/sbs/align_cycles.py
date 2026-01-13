from tifffile import imwrite

from lib.sbs.align_cycles import align_cycles
from lib.shared.io import read_image

# load image data (supports both TIFF and Zarr)
image_data = [read_image(file_path) for file_path in snakemake.input]

# align cycles
aligned_data = align_cycles(
    image_data,
    channel_order=snakemake.params.channel_names,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    window=snakemake.params.window,
    skip_cycles=snakemake.params.skip_cycles_indices,
    manual_background_cycle=snakemake.params.manual_background_cycle_index,
    manual_channel_mapping=snakemake.params.manual_channel_mapping,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
