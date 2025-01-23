import pandas as pd

from lib.merge.format_merge import (
    fov_distance,
    identify_single_gene_mappings,
    calculate_channel_mins,
)

# Load data for formatting merge data
merge_data = pd.read_hdf(snakemake.input[0])
sbs_cells = pd.read_hdf(snakemake.input[1])
phenotype_min_cp = pd.read_hdf(snakemake.input[2])

# Add FOV distances for both imaging modalities
print("Adding FOV distances...")
merge_data = merge_data.pipe(
    fov_distance, i="i_0", j="j_0", dimensions=(2960, 2960), suffix="_0"
)
merge_data = merge_data.pipe(
    fov_distance, i="i_1", j="j_1", dimensions=(1480, 1480), suffix="_1"
)

# Identify single gene mappings in SBS
sbs_cells["mapped_single_gene"] = sbs_cells.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)
# Merge cell information from sbs
print("Merging gene mapping information from sbs...")
merge_data = merge_data.merge(
    sbs_cells[
        ["well", "tile", "cell", "sgRNA_0", "gene_symbol_0", "mapped_single_gene"]
    ].rename({"tile": "site", "cell": "cell_1"}, axis=1),
    how="left",
    on=["well", "site", "cell_1"],
)

# Calculate minimum channel values for cells
phenotype_min_cp = calculate_channel_mins(phenotype_min_cp)
# Merge cell information from ph
print("\nMerging cell intensity information from ph data...")
merge_data = merge_data.merge(
    phenotype_min_cp[["well", "tile", "label", "channels_min"]].rename(
        columns={"label": "cell_0"}
    ),
    how="left",
    on=["well", "tile", "cell_0"],
)

# Save formatted merge data
merge_data.to_hdf(snakemake.output[0], "x", mode="w")
