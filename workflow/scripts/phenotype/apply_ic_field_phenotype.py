from lib.shared.illumination_correction import apply_ic_field
from lib.shared.io import read_image, save_image

# Load raw image data
raw_image_data = read_image(snakemake.input[0])

# Load illumination correction field
ic_field = read_image(snakemake.input[1])

# Apply illumination correction field
corrected_image_data = apply_ic_field(raw_image_data, correction=ic_field)

# Save corrected image data
save_image(corrected_image_data, snakemake.output[0])
