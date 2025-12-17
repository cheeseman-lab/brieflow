from tifffile import imread, imwrite

from lib.sbs.align_cycles import align_cycles

# Build tile identifier for log traceability
tile_id = f"P-{snakemake.wildcards.plate}_W-{snakemake.wildcards.well}_T-{snakemake.wildcards.tile}"

# load image data
image_data = [imread(file_path) for file_path in snakemake.input]

# Get manual cycle offsets if provided (passed as params from rule)
manual_cycle_offsets = getattr(snakemake.params, 'manual_cycle_offsets', None)

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
    manual_cycle_offsets=manual_cycle_offsets,
    verbose=True,
    tile_id=tile_id,
)

# Save the aligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
