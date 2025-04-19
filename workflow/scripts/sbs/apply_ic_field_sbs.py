from tifffile import imread, imwrite

from lib.shared.illumination_correction import apply_ic_field, combine_ic_images

# Load aligned image data
aligned_image_data = imread(snakemake.input[0])
aligned_image_data_segmentation_cycle = aligned_image_data[
    snakemake.params.segmentation_cycle_index
]

# Check if we're using a cycle other than the first one for segmentation
if snakemake.params.segmentation_cycle_index > 0:
    # Load IC field from first cycle (for extra channels)
    ic_field_first_cycle = imread(snakemake.input[1])
    # Load IC field from segmentation cycle (for base channels)
    ic_field_seg_cycle = imread(snakemake.input[2])
    # Combine the IC fields, using ALL extra channels from first cycle
    ic_field = combine_ic_images(
        [ic_field_first_cycle, ic_field_seg_cycle], 
        [snakemake.params.extra_channel_indices, None]
    )
else:
    # Just use the IC field from the first cycle
    ic_field = imread(snakemake.input[1])

# Apply illumination correction field
corrected_image_data = apply_ic_field(
    aligned_image_data_segmentation_cycle, correction=ic_field
)

# Save corrected image data
imwrite(snakemake.output[0], corrected_image_data)