import pandas as pd

from lib.aggregate.load_format_data import clean_cell_data

merged_final = pd.read_hdf(snakemake.input[0])

cleaned_data = clean_cell_data(
    merged_final, snakemake.params["population_feature"], filter_single_gene=False
)

del merged_final

tranformations = pd.read_csv(snakemake.input[1], sep="\t")
