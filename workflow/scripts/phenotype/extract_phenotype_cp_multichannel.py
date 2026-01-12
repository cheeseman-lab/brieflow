import numpy as np
from lib.shared.io import read_image

# load inputs
data_phenotype = read_image(snakemake.input[0])
nuclei = read_image(snakemake.input[1])
cells = read_image(snakemake.input[2])
cytoplasms = read_image(snakemake.input[3])

# Handle Z-dimension mismatch: if data is CZYX but labels are CYX, take mean projection
if data_phenotype.ndim == 4 and nuclei.ndim == 3:
    print(
        f"Taking mean projection: data_phenotype shape {data_phenotype.shape} -> ",
        end="",
    )
    # Take mean projection along Z axis (axis=1 for CZYX)
    data_phenotype = np.mean(data_phenotype, axis=1).astype(data_phenotype.dtype)
    print(f"{data_phenotype.shape}")
elif data_phenotype.ndim != nuclei.ndim:
    raise ValueError(
        f"Dimension mismatch: data_phenotype has {data_phenotype.ndim} dims "
        f"but nuclei has {nuclei.ndim} dims. Expected both to be 3D (CYX) or 4D (CZYX)."
    )

# Squeeze label images if they have a singleton channel dimension
# Labels should be 2D (YX) for regionprops, not 3D (CYX) with C=1
if nuclei.ndim == 3 and nuclei.shape[0] == 1:
    nuclei = np.squeeze(nuclei, axis=0)
if cells.ndim == 3 and cells.shape[0] == 1:
    cells = np.squeeze(cells, axis=0)
if cytoplasms.ndim == 3 and cytoplasms.shape[0] == 1:
    cytoplasms = np.squeeze(cytoplasms, axis=0)

if snakemake.params.cp_method == "cp_measure":
    from lib.phenotype.extract_phenotype_cp_measure import (
        extract_phenotype_cp_measure,
    )

    # TO-DO: Ensure conda environment is set up for cp_measure when using this method.
    # A quick guide:
    # 1. Clone brieflow environment:
    #    conda create --name brieflow_cpmeasure_env --clone brieflow_main_env
    # 2. Activate the environment:
    #    conda activate brieflow_cpmeasure_env
    # 3. Install the required package:
    #    pip install cp-measure
    # 4. Verify dependencies with 'conda list'- Cpmeasure requires Python 3.8 or later, and the following package versions:
    #    - NumPy 1.24.3*
    #    - centrosome 1.3.0*
    #    If you have issues running cpmeasure, you may need to downgrade these packages in the cloned environment.
    phenotype_cp = extract_phenotype_cp_measure(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        channel_names=snakemake.params.channel_names,
    )
else:
    from lib.phenotype.extract_phenotype_cp_multichannel import (
        extract_phenotype_cp_multichannel,
    )

    # extract phenotype CellProfiler information
    phenotype_cp = extract_phenotype_cp_multichannel(
        data_phenotype=data_phenotype,
        nuclei=nuclei,
        cells=cells,
        cytoplasms=cytoplasms,
        foci_channel=snakemake.params.foci_channel,
        channel_names=snakemake.params.channel_names,
        wildcards=snakemake.wildcards,
    )

# save phenotype cp
phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")
