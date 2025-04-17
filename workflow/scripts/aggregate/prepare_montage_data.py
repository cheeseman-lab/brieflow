from pathlib import Path
import pandas as pd

from lib.aggregate.montage_utils import add_filenames
from lib.shared.file_utils import get_filename

# Create output directory
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

# Load cell data
montage_columns = [
    "gene_symbol_0",
    "sgRNA_0",
    "plate",
    "well",
    "tile",
    "i_0",
    "j_0",
]
cell_data = pd.concat(
    [pd.read_parquet(p, columns=montage_columns) for p in snakemake.input],
    ignore_index=True,
)

# Prepare for montage
prepared_cell_data = add_filenames(cell_data, Path(snakemake.params.root_fp))

# Get combos of gene and sgrna
gene_sgrna_combos = (
    prepared_cell_data[["gene_symbol_0", "sgRNA_0"]].drop_duplicates().dropna()
)

# Save one file per gene/sgRNA combo
for _, row in gene_sgrna_combos.iterrows():
    print(f"Saving {row['gene_symbol_0']} {row['sgRNA_0']}...")
    gene = row["gene_symbol_0"]
    sgrna = row["sgRNA_0"]

    # Filter data for this gene/sgRNA combo
    subset = prepared_cell_data[
        (prepared_cell_data["gene_symbol_0"] == gene)
        & (prepared_cell_data["sgRNA_0"] == sgrna)
    ]

    # Save to TSV
    save_path = output_dir / get_filename(
        {"gene": gene, "sgrna": sgrna},
        "montage_data",
        "tsv",
    )

    subset.to_csv(save_path, sep="\t", index=False)

    if int(_) > 10:
        break
