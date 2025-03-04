from tifffile import imwrite

from lib.preprocess.preprocess import nd2_to_tiff

# convert the ND2 file to a TIF image array
image_array = nd2_to_tiff(
    snakemake.input, channel_order_flip=snakemake.params.channel_order_flip
)

# save TIF image array to the output path
imwrite(snakemake.output[0], image_array)
