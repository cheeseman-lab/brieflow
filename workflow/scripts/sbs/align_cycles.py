from tifffile import imread, imwrite

from lib.sbs.align_cycles import align_cycles

# load image data
image_data = [imread(file_path) for file_path in snakemake.input]

# align cycles with resize functionality
aligned_data = align_cycles(
    image_data,
    channel_order=snakemake.params.channel_names,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    skip_cycles=snakemake.params.skip_cycles_indices,
    manual_background_cycle=snakemake.params.manual_background_cycle_index,
    # New resize parameters
    resize_cycles=snakemake.params.resize_cycles,
    target_spatial_shape=snakemake.params.target_spatial_shape,
    resize_method=snakemake.params.resize_method,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
