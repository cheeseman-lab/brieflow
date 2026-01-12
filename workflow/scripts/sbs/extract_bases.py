from lib.sbs.extract_bases import extract_bases
from lib.shared.io import read_image


# load peaks data (supports TIFF and Zarr)
peaks_data = read_image(snakemake.input[0])

# load max filtered data
max_filtered_data = read_image(snakemake.input[1])

# load cells data
cells_data = read_image(snakemake.input[2])

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
