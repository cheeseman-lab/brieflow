from tifffile import imwrite

from lib.shared.illumination_correction import calculate_ic_field

# convert the ND2 file to a TIF image array
ic_field = calculate_ic_field(snakemake.input, threading=snakemake.params.threading)

# save TIF image array to the output path
imwrite(snakemake.output[0], ic_field)
