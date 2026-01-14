from tifffile import imwrite
from lib.shared.illumination_correction import apply_ic_field, combine_ic_images
from lib.shared.io import read_image

# Load aligned image data
aligned_image_data = imread(snakemake.input[0])

# Logic based on whether DAPI and CYTO come from same or different cycles
if snakemake.params.dapi_cycle != snakemake.params.cyto_cycle:
    # Different cycles - combine image data from both cycles
    aligned_image_data_segmentation_cycle = combine_ic_images(
        [
            aligned_image_data[snakemake.params.dapi_cycle_index],
            aligned_image_data[snakemake.params.cyto_cycle_index],
        ],
        [snakemake.params.extra_channel_indices, None],
    )
else:
    # Same cycle - use single cycle
    aligned_image_data_segmentation_cycle = aligned_image_data[
        snakemake.params.cyto_cycle_index
    ]

# Combine IC fields based on whether DAPI and CYTO come from same or different cycles
if snakemake.params.dapi_cycle != snakemake.params.cyto_cycle:
    # Different cycles - combine IC fields
    ic_field_dapi = read_image(snakemake.input[1])  # DAPI cycle IC
    ic_field_cyto = read_image(snakemake.input[2])  # CYTO cycle IC
    ic_field = combine_ic_images(
        [ic_field_dapi, ic_field_cyto], [snakemake.params.extra_channel_indices, None]
    )
else:
    # Same cycle - use one IC field (DAPI cycle IC)
    ic_field = read_image(snakemake.input[1])

# Apply illumination correction field
corrected_image_data = apply_ic_field(
    aligned_image_data_segmentation_cycle, correction=ic_field
)

# Save corrected image data
imwrite(snakemake.output[0], corrected_image_data)
