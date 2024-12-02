import pandas as pd
from joblib import Parallel, delayed


# Define function to read df tsv files
def get_file(f):
    try:
        return pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        pass


# Load, concatenate, and save the phenotype CellProfiler data
arr_reads = Parallel(n_jobs=snakemake.threads)(
    delayed(get_file)(file) for file in snakemake.input
)
phenotype_cp = pd.concat(arr_reads)
phenotype_cp.to_hdf(snakemake.output[0], "x", mode="w")

# Save subset of features
phenotype_cp_min = phenotype_cp[
    [
        "well",
        "tile",
        "label",
        "cell_i",
        "cell_j",
        "cell_bounds_0",
        "cell_bounds_1",
        "cell_bounds_2",
        "cell_bounds_3",
        "cell_dapi_min",
        "cell_cenpa_min",
        "cell_coxiv_min",
        "cell_wga_min",
    ]
]
phenotype_cp_min.to_hdf(snakemake.output[1], "x", mode="w")
