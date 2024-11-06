from skimage.io import imread

from shared.extract_phenotype_minimal import extract_phenotype_minimal

# load nuclei data
nuclei_data = imread(snakemake.input[0])

# extract minimal phenotype information
mininal_phenotype_data = extract_phenotype_minimal(
    phenotype_data=nuclei_data,
    nuclei_data=nuclei_data,
    wildcards=snakemake.wildcards,
)

# save minimal phenotype data
mininal_phenotype_data.to_csv(snakemake.output[0], index=False, sep="\t")
