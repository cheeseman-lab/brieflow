from pathlib import Path
import multiprocessing

import pyarrow.dataset as ds
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.montage_utils import add_filenames
from lib.shared.file_utils import get_filename


# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

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
prepared_cell_data = add_filenames(cell_data, Path(snakemake.params.root_fp))

# Group rows by gene + sgRNA
gene_sgrna_groups = prepared_cell_data.groupby(
    ["gene_symbol_0", "cell_barcode_0"], sort=False
)

print(f"Saving {gene_sgrna_groups.ngroups} gene/sgRNA combos to {output_dir}")
print(f"Using {multiprocessing.cpu_count()} CPUs")


def write_group(name_group):
    (gene, sgrna), df = name_group

    print(f"Processing {gene}, {sgrna}")

    df.to_csv(
        output_dir
        / get_filename({"gene": gene, "sgrna": sgrna}, "montage_data", "tsv"),
        sep="\t",
        index=False,
    )


with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
    ex.map(write_group, gene_sgrna_groups)