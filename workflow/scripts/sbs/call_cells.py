import pandas as pd

from lib.sbs.call_cells import call_cells

# load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# load df_barcode_library
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep="\t")

# call cells
cells_data = call_cells(
    reads_data=reads_data,
    df_barcode_library=df_barcode_library,
    q_min=snakemake.params.q_min,
    barcode_col=snakemake.params.barcode_col,
    error_correct=snakemake.params.error_correct,
)

# save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
