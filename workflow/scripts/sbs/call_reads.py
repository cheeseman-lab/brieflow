import pandas as pd

from lib.sbs.call_reads import call_reads
from lib.shared.image_io import read_image

# Load bases data
bases_data = pd.read_csv(snakemake.input[0], sep="\t")

# Load peaks data
peaks_data = read_image(snakemake.input[1])

# Load the barcode codebook only for the library-aware decoder
codebook = None
if snakemake.params.call_reads_method == "merfish":
    if snakemake.params.error_correct:
        raise ValueError(
            "sbs.error_correct must be false when call_reads_method='merfish'."
        )
    if not snakemake.params.codebook_fp:
        raise ValueError("method='merfish' requires 'sbs.df_barcode_library_fp'.")
    codebook = pd.read_csv(snakemake.params.codebook_fp, sep="\t")

# Call reads
reads_data = call_reads(
    bases_data=bases_data,
    peaks_data=peaks_data,
    method=snakemake.params.call_reads_method,
    chemistry=snakemake.params.chemistry,
    combinatorial=snakemake.params.combinatorial,
    codebook=codebook,
)

# Save reads data
reads_data.to_csv(snakemake.output[0], index=False, sep="\t")
