import sys
import traceback
from lib.shared.io import read_image, save_image, read_pixel_size
from lib.shared.illumination_correction import apply_ic_field

try:
    # load raw image data
    raw_image_data = read_image(snakemake.input[0])
    print(f"Raw image shape: {raw_image_data.shape}", file=sys.stderr)

    # Read pixel size from input image
    pixel_size_z, pixel_size_y, pixel_size_x = read_pixel_size(snakemake.input[0])

    # load illumination correction field
    ic_field = read_image(snakemake.input[1])
    print(f"IC field shape: {ic_field.shape}", file=sys.stderr)

    # apply illumination correction field
    corrected_image_data = apply_ic_field(raw_image_data, correction=ic_field)

    # save corrected image data
    save_image(
        corrected_image_data,
        snakemake.output[0],
        pixel_size_z=pixel_size_z,
        pixel_size_y=pixel_size_y,
        pixel_size_x=pixel_size_x,
        channel_names=snakemake.params.channel_names,
    )
except Exception as e:
    with open("apply_ic_error.log", "w") as f:
        f.write(f"Error in apply_ic_field_phenotype: {e}\n")
        traceback.print_exc(file=f)
    sys.exit(1)
