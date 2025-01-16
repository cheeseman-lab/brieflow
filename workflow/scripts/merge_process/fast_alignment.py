# import pandas as pd

# from lib.merge.hash import hash_process_info

print(snakemake.input)
print("Done!")

# # Load dfs with metadta on well level
# phenotype_metadata = pd.read_csv(snakemake.input[0], sep="\t")
# sbs_metadata = pd.read_csv(snakemake.input[1], sep="\t")
# # Load phentoype/sbs info on plate level
# phenotype_info = pd.read_hdf(snakemake.input[2], sep="\t")
# sbs_info = pd.read_hdf(snakemake.input[3], sep="\t")

# # Derive alignment hashes
# phenotype_info_hash = hash_process_info(phenotype_info)
# sbs_info_hash = hash_process_info(sbs_info)

# # Read XY coordinates for phenotype and SBS
# phenotype_xy = (
#     pd.read_pickle(phenotype_metadata)
#     .rename(columns={"field_of_view": "tile", "x_data": "x", "y_data": "y"})
#     .set_index("tile")[["x", "y"]]
# )

# sbs_xy = (
#     pd.read_pickle(sbs_metadata)
#     .rename(columns={"field_of_view": "tile", "x_data": "x", "y_data": "y"})
#     .set_index("tile")[["x", "y"]]
# )
