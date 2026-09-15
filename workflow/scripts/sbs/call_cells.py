"""Call cells from sequencing reads using unified interface.

Supports both single-barcode and multi-barcode protocols.
"""

from lib.sbs.call_cells import call_cells, load_barcode_library
from lib.shared.parquet_io import read_table, write_parquet

# Get configuration from params
params = snakemake.params.config

# Load reads data (parquet in a fresh DAG; read_table resolves parquet-or-tsv)
reads_data = read_table(snakemake.input[0])

# Load barcode library
df_barcode_library = load_barcode_library(params["df_barcode_library_fp"])

# Choose calling method based on barcode_type parameter
barcode_type = params.get("barcode_type", "simple")

if barcode_type == "multi":
    cells_data = call_cells(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        map_start=params["map_start"],
        map_end=params["map_end"],
        prefix_map=params["prefix_map"],
        recomb_start=params["recomb_start"],
        recomb_end=params["recomb_end"],
        prefix_recomb=params["prefix_recomb"],
        recomb_filter_col=params["recomb_filter_col"],
        recomb_q_thresh=params["recomb_q_thresh"],
        error_correct=params["error_correct"],
        sort_calls=params["sort_calls"],
        max_distance=params["max_distance"],
        n_barcodes=params["n_barcodes"],
        barcode_info_cols=params["barcode_info_cols"],
    )
else:
    cells_data = call_cells(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        barcode_col=params["barcode_col"],
        prefix_col=params["prefix_col"],
        error_correct=params["error_correct"],
        sort_calls=params["sort_calls"],
        max_distance=params["max_distance"],
        n_barcodes=params["n_barcodes"],
    )

# Save cells data
write_parquet(cells_data, snakemake.output[0])
