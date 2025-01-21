import pandas as pd

from lib.merge.hash import hash_process_info, multistep_alignment

print(snakemake.input)
print("Done!")


# Load dfs with metadata on well level
phenotype_metadata = pd.read_hdf(snakemake.input[0])
sbs_metadata = pd.read_hdf(snakemake.input[1])
# Load phentoype/sbs info on plate level
phenotype_info = pd.read_hdf(snakemake.input[2])
sbs_info = pd.read_hdf(snakemake.input[3])

print(phenotype_metadata, sbs_metadata, phenotype_info, sbs_info)

# Derive fast alignment per well
well_alignments = []

wells = phenotype_metadata["well"].unique().tolist()
print(wells)
for well in wells:
    print(f"Processing well {well}...")
    # Load well subsets of data
    well_phenotype_metadata = phenotype_metadata[phenotype_metadata["well"] == well]
    well_sbs_metadata = sbs_metadata[sbs_metadata["well"] == well]
    well_phenotype_info = phenotype_info[phenotype_info["well"] == well]
    well_sbs_info = sbs_info[sbs_info["well"] == well]

    # Format XY coordinates for phenotype and SBS
    phenotype_xy = well_phenotype_metadata.rename(
        columns={"x_pos": "x", "y_pos": "y"}
    ).set_index("tile")[["x", "y"]]
    sbs_xy = well_sbs_metadata.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index(
        "tile"
    )[["x", "y"]]

    # Hash phenotype and sbs info
    phenotype_info_hash = hash_process_info(well_phenotype_info)
    print("Hashed phenotype info!")
    sbs_info_hash = hash_process_info(well_sbs_info)
    print("Hashed SBS info!")

    well_alignment = multistep_alignment(
        phenotype_info_hash,
        sbs_info_hash,
        phenotype_xy,
        sbs_xy,
        det_range=snakemake.params.det_range,
        score=snakemake.params.score,
        initial_sites=snakemake.params.initial_sites,
        n_jobs=snakemake.threads,
    )
    well_alignments.append(well_alignment)
    print("Well alignment completed!")

well_alignments.to_hdf(snakemake.output[0], "x", mode="w")
