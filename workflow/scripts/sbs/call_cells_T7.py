import pandas as pd
from pandas.errors import EmptyDataError
from lib.sbs.call_cells_T7 import call_cells_T7, prep_T7_reads

# Load reads data
try:
    reads_data = pd.read_csv(snakemake.input[0], sep="\t")
except EmptyDataError:
    reads_data = pd.DataFrame()

# Load and process df_pool
df_design = pd.read_csv(snakemake.params.df_design_path, index_col=None)
df_pool = df_design.drop(columns=['Unnamed: 0'], errors='ignore').rename(columns={'target':'gene_symbol'})
df_pool['prefix_map'] = df_pool['iBAR2_f7']

# Prepare T7 reads
df_reads = prep_T7_reads(
    reads_data,
    map_col=snakemake.params.map_col,
)

try:
    df_reads['prefix_map'] = df_reads['barcode']
except KeyError:
    # no 'barcode' column ⇒ treat as no reads at all
    df_reads = pd.DataFrame()

    
# Call cells
cells_data = call_cells_T7(
    reads_data=df_reads,
    df_pool=df_pool,
    q_min=snakemake.params.q_min,
    map_col=snakemake.params.map_col,
    error_correct=snakemake.params.error_correct,
    barcode_info_cols=snakemake.params.barcode_info_cols,
    max_distance=snakemake.params.max_distance
)

# Save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")