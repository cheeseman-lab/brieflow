from skimage.io import imsave

from lib.shared.file_utils import read_stack
from lib.shared.illumination_correction import apply_ic_field

# load raw image data
raw_image_data = read_stack(snakemake.input[0])

# load illumination correction field
ic_field = read_stack(snakemake.input[1])

# apply illumination correction field
corrected_image_data = apply_ic_field(raw_image_data, correction=ic_field)

# save corrected image data
imsave(snakemake.output[0], corrected_image_data)