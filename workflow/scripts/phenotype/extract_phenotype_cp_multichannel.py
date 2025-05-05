from tifffile import imread

from lib.phenotype.extract_phenotype_cp_multichannel import (
    extract_phenotype_cp_multichannel,
)

from lib.phenotype.extract_phenotype_cp_measure import extract_phenotype_cp_measure

# load inputs
data_phenotype = imread(snakemake.input[0])
nuclei = imread(snakemake.input[1])
cells = imread(snakemake.input[2])
cytoplasms = imread(snakemake.input[3])

if snakemake.params.cp_method == "cp_measure":
    # Extract features using cp_measure
    phenotype_cp = extract_phenotype_cp_measure(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        channel_names=snakemake.params.channel_names,
    )
else:
    # extract phenotype CellProfiler information
    phenotype_cp = extract_phenotype_cp_multichannel(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        foci_channel=snakemake.params.foci_channel,
        channel_names=snakemake.params.channel_names,
        wildcards=snakemake.wildcards,
    )

# save phenotype cp
phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")
