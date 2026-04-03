from lib.shared.extract_phenotype_minimal import extract_phenotype_minimal
from lib.shared.io import read_image

# Load nuclei data
nuclei_data = read_image(snakemake.input[0])

# Build wildcards dict, synthesizing 'well' from 'row'+'col' in zarr mode
wc = dict(snakemake.wildcards)
if "row" in wc and "col" in wc and "well" not in wc:
    wc["well"] = wc["row"] + wc["col"]

# Extract minimal phenotype information
phenotype_minimal = extract_phenotype_minimal(
    phenotype_data=nuclei_data,
    nuclei_data=nuclei_data,
    wildcards=wc,
)

# save minimal phenotype data
phenotype_minimal.to_csv(snakemake.output[0], index=False, sep="\t")
