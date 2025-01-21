import pandas as pd

from lib.shared.file_utils import parse_filename
from lib.merge.hash import hash_process_info

print(snakemake.input)
print(snakemake.input.phenotype_metadata)
print(snakemake.input.sbs_metadata)
print("Done!")


# Load dfs with metadata on well level
phenotype_metadata = pd.read_csv(snakemake.input.phenotype_metadata, sep="\t")
sbs_metadata = pd.read_csv(snakemake.input.sbs_metadata, sep="\t")
# Load phentoype/sbs info on plate level
phenotype_info = pd.read_hdf(snakemake.input.phenotype_info, sep="\t")
sbs_info = pd.read_hdf(snakemake.input.sbs_info, sep="\t")

# Derive alignment hashes
phenotype_info_hash = hash_process_info(phenotype_info)
sbs_info_hash = hash_process_info(sbs_info)

# Read XY coordinates for phenotype and SBS
phenotype_xy = phenotype_metadata.rename(
    columns={"field_of_view": "tile", "x_data": "x", "y_data": "y"}
).set_index("tile")[["x", "y"]]

sbs_xy = sbs_metadata.rename(
    columns={"field_of_view": "tile", "x_data": "x", "y_data": "y"}
).set_index("tile")[["x", "y"]]

# Derive fast alignment per well
well_fast_alignments = []

for index, phenotype_metadata_fp in enumerate(snakemake.input.phenotype_metadata):
    # get well of metadata file
    well = parse_filename(phenotype_metadata_fp)[0]
