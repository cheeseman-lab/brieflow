from lib.preprocess.preprocess import extract_tile_metadata

# Extract metadata from ND2 file paths using the _extract_metadata_tile function
cycle = getattr(snakemake.params, "cycle", None)
metadata_df = extract_tile_metadata(
    snakemake.input[0],
    snakemake.params.plate,
    snakemake.params.well,
    snakemake.params.tile,
    cycle,
)

# Save the extracted metadata to the output path
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
