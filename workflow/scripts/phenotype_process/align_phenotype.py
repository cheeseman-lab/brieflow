from tifffile import imread, imwrite

from lib.phenotype_process.align_channels import align_phenotype_channels

# load image data
image_data = imread(snakemake.input[0])

# align based on parameter
if snakemake.params.align:
    print("Aligning channels...")
    aligned_data = align_phenotype_channels(
        image_data,
        target=snakemake.params.target,
        source=snakemake.params.source,
        riders=snakemake.params.riders,
        remove_channel=snakemake.params.remove_channel,
    )
else:
    print("Skipping alignment...")
    aligned_data = image_data

# Save the aligned/unaligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)