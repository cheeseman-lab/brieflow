import pandas as pd

from lib.sbs.call_cells import call_cells
from lib.shared.file_utils import read_csv_gcs_compatible

# Get GCS project from config
gcs_project = snakemake.config["all"].get("gcs_project")

# load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# load df_barcode_library
df_barcode_library = read_csv_gcs_compatible(snakemake.params.df_barcode_library_fp, gcs_project=gcs_project, sep="\t")

# call cells
cells_data = call_cells(
    reads_data=reads_data,
    df_barcode_library=df_barcode_library,
    q_min=snakemake.params.q_min,
    barcode_col=snakemake.params.barcode_col,
    prefix_col=snakemake.params.prefix_col,
    error_correct=snakemake.params.error_correct,
    sort_calls=snakemake.params.sort_calls,
)

# save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
