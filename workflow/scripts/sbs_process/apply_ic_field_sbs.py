from tifffile import imread, imwrite

from lib.shared.illumination_correction import apply_ic_field, combine_ic_images

# load aligned image data
aligned_image_data = imread(snakemake.input[0])
aligned_image_data_segmentation_cycle = aligned_image_data[
    snakemake.params.segmentation_cycle_index
]

# load ic field
if snakemake.params.keep_extras:
   ic_field_dapi = imread(snakemake.input[1])
   ic_field_full = imread(snakemake.input[2])
   ic_field = combine_ic_images([ic_field_dapi, ic_field_full], [snakemake.params.dapi_index, None])
else:
   ic_field = imread(snakemake.input[2])

# apply illumination correction field
corrected_image_data = apply_ic_field(
    aligned_image_data_segmentation_cycle, correction=ic_field
)

# save corrected image data
imwrite(snakemake.output[0], corrected_image_data)
