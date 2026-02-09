from lib.sbs.extract_bases import extract_bases
from lib.shared.io import read_image


# Load peaks data
peaks_data = read_image(snakemake.input[0])

# Load max filtered data
max_filtered_data = read_image(snakemake.input[1])

# Load cells data
cells_data = read_image(snakemake.input[2])

# Extract bases
# Build wildcards dict, synthesizing 'well' from 'row'+'col' in zarr mode
wc = dict(snakemake.wildcards)
if "row" in wc and "col" in wc and "well" not in wc:
    wc["well"] = wc["row"] + wc["col"]

bases_data = extract_bases(
    peaks_data=peaks_data,
    max_filtered_data=max_filtered_data,
    cells_data=cells_data,
    threshold_peaks=snakemake.params.threshold_peaks,
    bases=snakemake.params.bases,
    wildcards=wc,
)

# Save bases data
bases_data.to_csv(snakemake.output[0], index=False, sep="\t")
