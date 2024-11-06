from skimage.io import imread

from lib.sbs_process.extract_bases import extract_bases


# load peaks data
peaks_data = imread(snakemake.input[0])

# load max filtered data
max_filtered_data = imread(snakemake.input[1])

# load cells data
cells_data = imread(snakemake.input[2])

# extract bases
bases_data = extract_bases(
    peaks_data=peaks_data,
    max_filtered_data=max_filtered_data,
    cells_data=cells_data,
    threshold_peaks=snakemake.params.threshold_peaks,
    bases=snakemake.params.bases,
    wildcards=dict(snakemake.wildcards),
)

# save bases data
bases_data.to_csv(snakemake.output[0], index=False, sep="\t")
