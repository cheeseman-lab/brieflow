"""Call cells from sequencing reads using unified interface.

Supports both single-barcode and multi-barcode protocols.
"""

import pandas as pd

from lib.sbs.call_cells import call_cells

# Get configuration from params
params = snakemake.params.config

# Load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load barcode library
df_barcode_library = pd.read_csv(params["df_barcode_library_fp"], sep="\t")

# Choose calling method based on barcode_type parameter
barcode_type = params.get("barcode_type", "simple")

if barcode_type == "multi":
    cells_data = call_cells(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        map_start=params["map_start"],
        map_end=params["map_end"],
        map_col=params["map_col"],
        recomb_start=params["recomb_start"],
        recomb_end=params["recomb_end"],
        recomb_col=params["recomb_col"],
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
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
