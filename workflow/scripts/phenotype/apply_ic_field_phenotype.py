from tifffile import imwrite

from lib.shared.illumination_correction import apply_ic_field
from lib.shared.io import read_image

# load raw image data (supports TIFF and Zarr)
raw_image_data = read_image(snakemake.input[0])

# load illumination correction field
ic_field = read_image(snakemake.input[1])

# apply illumination correction field
corrected_image_data = apply_ic_field(raw_image_data, correction=ic_field)

# save corrected image data
imwrite(snakemake.output[0], corrected_image_data)
