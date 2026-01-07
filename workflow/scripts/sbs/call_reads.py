import pandas as pd
from lib.shared.io import read_image

from lib.sbs.call_reads import call_reads

# load bases data
bases_data = pd.read_csv(snakemake.input[0], sep="\t")

# load peaks data
peaks_data = read_image(snakemake.input[1]).squeeze()

# call reads
reads_data = call_reads(
    bases_data=bases_data,
    peaks_data=peaks_data,
    method=snakemake.params.call_reads_method,
)

# save reads data
reads_data.to_csv(snakemake.output[0], index=False, sep="\t")
