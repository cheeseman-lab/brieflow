from lib.shared.io import read_image

# load inputs (supports TIFF and Zarr)
data_phenotype = read_image(snakemake.input[0])
nuclei = read_image(snakemake.input[1])
cells = read_image(snakemake.input[2])
cytoplasms = read_image(snakemake.input[3])

# Check if cell segmentation is enabled - if not, pass None to skip cell/cytoplasm features
segment_cells = snakemake.params.get("segment_cells", True)
if not segment_cells:
    cells = None
    cytoplasms = None

cp_method = snakemake.params.cp_method

if cp_method == "cp_measure":
    from lib.phenotype.extract_phenotype_cp_measure import (
        extract_phenotype_cp_measure,
    )

    # extract phenotype features using cp_measure
    phenotype_cp = extract_phenotype_cp_measure(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        channel_names=snakemake.params.channel_names,
    )
elif cp_method == "cp_emulator":
    from lib.phenotype.extract_phenotype_cp_emulator import (
        extract_phenotype_cp_emulator,
    )

    # extract phenotype features using CellProfiler emulator
    phenotype_cp = extract_phenotype_cp_emulator(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        foci_channel=snakemake.params.foci_channel_index,
        channel_names=snakemake.params.channel_names,
        wildcards=snakemake.wildcards,
    )
else:
    raise ValueError(
        f"Unknown cp_method: {cp_method}. Choose 'cp_measure' or 'cp_emulator'."
    )

# save phenotype cp
phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")
