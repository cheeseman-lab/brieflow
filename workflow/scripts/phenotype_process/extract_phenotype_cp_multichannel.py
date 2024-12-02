from lib.shared.file_utils import read_stack
from lib.phenotype_process.extract_phenotype_cp_multichannel import (
    extract_phenotype_cp_multichannel,
)

# load inputs
data_phenotype = read_stack(snakemake.input[0])
nuclei = read_stack(snakemake.input[1])
cells = read_stack(snakemake.input[2])
cytoplasms = read_stack(snakemake.input[3])

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
