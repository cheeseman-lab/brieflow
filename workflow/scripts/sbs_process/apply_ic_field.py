from lib.shared.illumination_correction import apply_ic_field
from skimage.io import imread, imsave

# load aligned image data
aligned_image_data = imread(snakemake.input[0])
aligned_image_data_segmentation_cycle = aligned_image_data[
    snakemake.params.segmentation_cycle_index - 1
]

# load illumination correction field
ic_field = imread(snakemake.input[1])
print(snakemake.input[1])
print(ic_field.shape)

# apply illumination correction field
corrected_image_data = apply_ic_field(
    aligned_image_data_segmentation_cycle, correction=ic_field
)

# save corrected image data
imsave(snakemake.output[0], corrected_image_data)
