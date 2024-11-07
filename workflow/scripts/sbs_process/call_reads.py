from skimage.io import imread
import pandas as pd

from lib.sbs_process.call_reads import call_reads

# load bases data
bases_data = pd.read_csv(snakemake.input[0], sep="\t")

# load peaks data
peaks_data = imread(snakemake.input[1])

# call reads
reads_data = call_reads(
    bases_data=bases_data,
    peaks_data=peaks_data,
)

# save reads data
reads_data.to_csv(snakemake.output[0], index=False, sep="\t")
