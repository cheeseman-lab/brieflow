from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal
from lib.shared.io import read_image

# load nuclei data (supports TIFF and Zarr)
nuclei_data = read_image(snakemake.input[0])

# extract minimal phenotype information
phenotype_minimal = extract_phenotype_minimal(
    phenotype_data=nuclei_data,
    nuclei_data=nuclei_data,
    wildcards=snakemake.wildcards,
)

# save minimal phenotype data
phenotype_minimal.to_csv(snakemake.output[0], index=False, sep="\t")
