import pandas as pd
from lib.sbs.call_cells_t7 import call_cells_T7, prep_T7_reads

# Load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load and process df_pool
df_design = pd.read_csv(snakemake.params.df_design_path, index_col=None)
df_pool = df_design.drop(columns=['Unnamed: 0']).rename(columns={'target':'gene_symbol'})
# df_pool['prefix_map'] = df_pool['iBAR_2']
# df_pool['prefix_recomb'] = df_pool['iBAR_1'].str.slice(0,3)

# Prepare T7 reads
df_reads = prep_T7_reads(
    reads_data,
    map_start=snakemake.params.map_start,
    map_end=snakemake.params.map_end,
    recomb_start=snakemake.params.recomb_start,
    recomb_end=snakemake.params.recomb_end,
    map_col=snakemake.params.map_col,
    recomb_col=snakemake.params.recomb_col
)

# Call cells
cells_data = call_cells_T7(
    reads_data=df_reads,
    df_pool=df_pool,
    q_min=snakemake.params.q_min,
    map_col=snakemake.params.map_col,
    recomb_col=snakemake.params.recomb_col,
    recomb_filter_col=snakemake.params.recomb_filter_col,
    recomb_q_thresh=snakemake.params.recomb_q_thresh,
    error_correct=snakemake.params.error_correct,
    barcode_info_cols=snakemake.params.barcode_info_cols,
    max_distance=snakemake.params.max_distance
)

# Save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")