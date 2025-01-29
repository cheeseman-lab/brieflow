import pandas as pd

from lib.aggregate.montage_utils import create_cell_montage

print("Generating montage for gene: ", snakemake.wildcards.gene)

# read cell data
cell_data = pd.read_hdf(input[0])

# subset data to inly include target gene and sgrna combination
cell_data = cell_data[
    (cell_data["gene"] == snakemake.wildcards.gene)
    & (cell_data["sgrna"] == snakemake.wildcards.sgrna)
]

montage = create_cell_montage(cell_data, snakemake.params.channels)

if montage is not None:
    save(
        output[0],
        montage,
        display_mode="grayscale",
        display_ranges=DISPLAY_RANGES[wildcards.channel],
    )
else:
    # Create empty file if no cells found
    open(output[0], "w").close()
