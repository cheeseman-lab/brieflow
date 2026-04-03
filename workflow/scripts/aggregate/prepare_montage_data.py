from pathlib import Path
import multiprocessing

import pyarrow.dataset as ds
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.montage_utils import add_filenames
from lib.shared.file_utils import get_filename


# Validate required params
if getattr(snakemake.params, "root_fp", None) is None:
    raise ValueError("Required config parameter 'root_fp' is not set")

# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

# Handle empty input gracefully
cell_check = ds.dataset(snakemake.input, format="parquet")
if cell_check.count_rows() == 0:
    print("WARNING: No cells in input, skipping montage data preparation")
    exit(0)

# Load cell data
montage_columns = [
    "gene_symbol_0",
    "cell_barcode_0",
    "plate",
    "well",
    "tile",
    "i_0",
    "j_0",
]

cell_data = ds.dataset(snakemake.input, format="parquet")
cell_data = cell_data.to_table(columns=montage_columns, use_threads=True)
cell_data = cell_data.to_pandas()

# Prepare for montage
img_fmt = snakemake.params.get("img_fmt", "tiff")
prepared_cell_data = add_filenames(
    cell_data, Path(snakemake.params.root_fp), img_fmt=img_fmt
)

# Group rows by gene + cell barcode
gene_barcode_groups = prepared_cell_data.groupby(
    ["gene_symbol_0", "cell_barcode_0"], sort=False
)

print(f"Saving {gene_barcode_groups.ngroups} gene/barcode combos to {output_dir}")
print(f"Using {multiprocessing.cpu_count()} CPUs")


def write_group(name_group):
    (gene, barcode), df = name_group

    print(f"Processing {gene}, {barcode}")

    df.to_csv(
        output_dir
        / get_filename({"gene": gene, "sgrna": barcode}, "montage_data", "tsv"),
        sep="\t",
        index=False,
    )


with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
    ex.map(write_group, gene_barcode_groups)
