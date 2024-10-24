import snakemake
from lib.preprocess.preprocess import nd2_to_tif
from skimage.io import imsave

# convert the ND2 file to a TIF image array
image_array = nd2_to_tif(snakemake.input[0], channel_order_flip=snakemake.params.channel_order_flip)

# save TIF image array to the output path
imsave(snakemake.output[0], image_array)
