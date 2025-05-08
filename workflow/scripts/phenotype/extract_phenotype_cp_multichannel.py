from tifffile import imread

# load inputs
data_phenotype = imread(snakemake.input[0])
nuclei = imread(snakemake.input[1])
cells = imread(snakemake.input[2])
cytoplasms = imread(snakemake.input[3])

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
