from tifffile import imread, imwrite

from lib.sbs.find_peaks import find_peaks

# load standard deviation data
standard_deviation_data = imread(snakemake.input[0])

# find peaks
peaks = find_peaks(
    standard_deviation_data=standard_deviation_data,
    width=snakemake.params.width,
)

# save peak data
imwrite(snakemake.output[0], peaks)
