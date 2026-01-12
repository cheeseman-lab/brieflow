from lib.shared.io import read_image, save_image
from lib.sbs.align_cycles import align_cycles


def read_and_project(file_path):
    img = read_image(file_path)
    # If 4D (C, Z, Y, X), max project along Z (axis 1)
    if img.ndim == 4:
        return img.max(axis=1)
    return img


# load image data
image_data = [read_and_project(file_path) for file_path in snakemake.input]

# align cycles
aligned_data = align_cycles(
    image_data,
    channel_order=snakemake.params.channel_names,
    method=snakemake.params.method,
    upsample_factor=snakemake.params.upsample_factor,
    skip_cycles=snakemake.params.skip_cycles_indices,
    manual_background_cycle=snakemake.params.manual_background_cycle_index,
)

# Save the aligned data
save_image(
    aligned_data,
    snakemake.output[0],
    pixel_size_z=snakemake.params.pixel_size_z,
    pixel_size_y=snakemake.params.pixel_size_y,
    pixel_size_x=snakemake.params.pixel_size_x,
    channel_names=snakemake.params.channel_names,
)
