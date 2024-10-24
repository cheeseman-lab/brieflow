import snakemake
from lib.shared.illumination_correction import calculate_ic_field
from skimage.io import imsave

# convert the ND2 file to a TIF image array
ic_field = calculate_ic_field(snakemake.input, threading=snakemake.params.threading)

# save TIF image array to the output path
imsave(snakemake.output[0], ic_field)
