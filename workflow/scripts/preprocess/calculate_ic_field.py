import sys
import traceback
from lib.shared.io import save_image, read_pixel_size
from lib.shared.illumination_correction import calculate_ic_field

try:
    # Read pixel size from the first input image
    pixel_size_z, pixel_size_y, pixel_size_x = read_pixel_size(snakemake.input[0])

    # convert the ND2 file to a TIF image array
    ic_field = calculate_ic_field(
        snakemake.input,
        threading=snakemake.params.threading,
        sample_fraction=snakemake.params.sample_fraction,
        smooth=snakemake.params.smooth,
    )

    # save TIF image array to the output path
    save_image(
        ic_field,
        snakemake.output[0],
        pixel_size_z=pixel_size_z,
        pixel_size_y=pixel_size_y,
        pixel_size_x=pixel_size_x,
        channel_names=snakemake.params.channel_names,
    )
except Exception as e:
    with open("calculate_ic_error.log", "w") as f:
        f.write(f"Error in calculate_ic_field.py: {e}\n")
        traceback.print_exc(file=f)
    sys.exit(1)
