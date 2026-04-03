from tifffile import imread

from lib.sbs.extract_bases import extract_bases


# Validate required params
for _param_name in ["threshold_peaks", "bases"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

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
