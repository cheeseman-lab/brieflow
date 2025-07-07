import pandas as pd
from lib.sbs.call_cells_multi import call_cells_multi, prep_multi_reads

# Load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# load df_barcode_library
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep="\t")

# Prepare multi reads
df_reads = prep_multi_reads(
    reads_data,
    map_start=snakemake.params.map_start,
    map_end=snakemake.params.map_end,
    recomb_start=snakemake.params.recomb_start,
    recomb_end=snakemake.params.recomb_end,
    map_col=snakemake.params.map_col,
    recomb_col=snakemake.params.recomb_col,
)

# Call cells
cells_data = call_cells_multi(
    reads_data=df_reads,
    df_barcode_library=df_barcode_library,
    q_min=snakemake.params.q_min,
    map_col=snakemake.params.map_col,
    recomb_col=snakemake.params.recomb_col,
    recomb_filter_col=snakemake.params.recomb_filter_col,
    recomb_q_thresh=snakemake.params.recomb_q_thresh,
    error_correct=snakemake.params.error_correct,
    barcode_info_cols=snakemake.params.barcode_info_cols,
    max_distance=snakemake.params.max_distance,
)

# Save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
