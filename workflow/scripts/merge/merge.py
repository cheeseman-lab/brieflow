import pandas as pd

from lib.merge.merge import merge_triangle_hash

# Load phenotype and sbs info with cell locations
phenotype_info = pd.read_hdf(snakemake.input[0])
sbs_info = pd.read_hdf(snakemake.input[1])

# Load alignment data
fast_alignment = pd.read_hdf(snakemake.input[2])

# Filter alignment data based on parameters
fast_alignment_filtered = fast_alignment[
    (fast_alignment["determinant"] >= snakemake.params.det_range[0])
    & (fast_alignment["determinant"] <= snakemake.params.det_range[1])
    & (fast_alignment["score"] > snakemake.params.score)
]

# Merge cells across all wells
merge_data = []
for index, alignment_row in fast_alignment_filtered.iterrows():
    # Determine wells, tiles, and sites for merging
    well = alignment_row["well"]
    phenotype_tile = alignment_row["tile"]
    sbs_site = alignment_row["site"]

    # Filter phenotype and sbs info to the relevant well and tile for merging
    phenotype_info_filtered = phenotype_info[
        (phenotype_info["well"] == well) & (phenotype_info["tile"] == phenotype_tile)
    ]
    sbs_info_filtered = sbs_info[
        (sbs_info["well"] == well) & (sbs_info["tile"] == sbs_site)
    ]

    # Merge cells for row of alignment data
    alignment_row_merge = merge_triangle_hash(
        phenotype_info_filtered,
        sbs_info_filtered,
        alignment_row,
        threshold=snakemake.params.threshold,
    )
    merge_data.append(alignment_row_merge)

# Compile and save merge data
merge_data = pd.concat(merge_data, ignore_index=True)
merge_data.to_hdf(snakemake.output[0], "x", mode="w")
