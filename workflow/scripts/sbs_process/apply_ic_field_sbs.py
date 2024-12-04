from tifffile import imread, imwrite

from lib.shared.illumination_correction import apply_ic_field

# load aligned image data
aligned_image_data = imread(snakemake.input[0])
aligned_image_data_segmentation_cycle = aligned_image_data[
    snakemake.params.segmentation_cycle_index - 1
]

# load illumination correction field
ic_field = imread(snakemake.input[1])

# apply illumination correction field
corrected_image_data = apply_ic_field(
    aligned_image_data_segmentation_cycle, correction=ic_field
)

# save corrected image data
imwrite(snakemake.output[0], corrected_image_data)
