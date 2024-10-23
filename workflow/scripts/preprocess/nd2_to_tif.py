from lib.preprocess.preprocess import nd2_to_tif
from lib.shared.io import save_tif

# convert the ND2 file to a TIF image array
image_array = nd2_to_tif(snakemake.input[0], channel_order_flip=snakemake.params.channel_order_flip)

# save TIF image array to the output path
save_tif(image_array, snakemake.output[0])
