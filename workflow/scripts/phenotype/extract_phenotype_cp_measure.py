from tifffile import imread

from lib.phenotype.extract_phenotype_cp_measure import extract_phenotype_cp_measure

# Load inputs, order: data_phenotype, nuclei, cells, cytoplasms -- aligning w exiting workflow
data_phenotype = imread(snakemake.input[0])
nuclei = imread(snakemake.input[1])
cells = imread(snakemake.input[2])
cytoplasms = imread(snakemake.input[3])

# Extract features using cp_measure
phenotype_cp = extract_phenotype_cp_measure(
    data_phenotype=data_phenotype,
    nuclei=nuclei,
    cells=cells,
    cytoplasms=cytoplasms,
    channel_names=snakemake.params.channel_names,
)

# Save results in TSV format, matching output of existing workflow
phenotype_cp.to_csv(snakemake.output[0], index=False, sep='\t')
print(snakemake.input)