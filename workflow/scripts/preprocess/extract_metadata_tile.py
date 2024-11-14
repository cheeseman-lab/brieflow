from lib.preprocess.preprocess import extract_metadata_tile

print(snakemake.input)

# Extract metadata from ND2 file paths using the _extract_metadata_tile function
metadata_df = extract_metadata_tile(
    snakemake.input, z_interval=snakemake.params.z_interval
)

# Save the extracted metadata to the output path
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
