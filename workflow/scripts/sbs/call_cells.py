"""Call cells from sequencing reads using unified interface.

Supports both single-barcode and multi-barcode protocols.
"""

import pandas as pd

from lib.sbs.call_cells import call_cells

# Load reads data
reads_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load barcode library
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep="\t")

# Build parameter dictionary from snakemake params
call_params = {
    "reads_data": reads_data,
    "df_barcode_library": df_barcode_library,
    "q_min": snakemake.params.q_min,
    "error_correct": snakemake.params.error_correct,
    "sort_calls": snakemake.params.sort_calls,
}

# Add single-barcode parameters if present
if hasattr(snakemake.params, "barcode_col"):
    call_params["barcode_col"] = snakemake.params.barcode_col
if hasattr(snakemake.params, "prefix_col") and snakemake.params.prefix_col:
    call_params["prefix_col"] = snakemake.params.prefix_col

# Add multi-barcode parameters if present
if hasattr(snakemake.params, "map_start") and snakemake.params.map_start:
    call_params["map_start"] = snakemake.params.map_start
if hasattr(snakemake.params, "map_end") and snakemake.params.map_end:
    call_params["map_end"] = snakemake.params.map_end
if hasattr(snakemake.params, "map_col") and snakemake.params.map_col:
    call_params["map_col"] = snakemake.params.map_col

# Add recombination parameters if present
if hasattr(snakemake.params, "recomb_start") and snakemake.params.recomb_start:
    call_params["recomb_start"] = snakemake.params.recomb_start
if hasattr(snakemake.params, "recomb_end") and snakemake.params.recomb_end:
    call_params["recomb_end"] = snakemake.params.recomb_end
if hasattr(snakemake.params, "recomb_col") and snakemake.params.recomb_col:
    call_params["recomb_col"] = snakemake.params.recomb_col
if (
    hasattr(snakemake.params, "recomb_filter_col")
    and snakemake.params.recomb_filter_col
):
    call_params["recomb_filter_col"] = snakemake.params.recomb_filter_col
if hasattr(snakemake.params, "recomb_q_thresh"):
    call_params["recomb_q_thresh"] = snakemake.params.recomb_q_thresh

# Add barcode info columns if present
if (
    hasattr(snakemake.params, "barcode_info_cols")
    and snakemake.params.barcode_info_cols
):
    call_params["barcode_info_cols"] = snakemake.params.barcode_info_cols

# Add max_distance if present
if hasattr(snakemake.params, "max_distance"):
    call_params["max_distance"] = snakemake.params.max_distance

# Call cells using unified function
cells_data = call_cells(**call_params)

# Save cells data
cells_data.to_csv(snakemake.output[0], index=False, sep="\t")
