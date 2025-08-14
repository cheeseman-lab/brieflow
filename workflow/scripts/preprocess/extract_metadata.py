"""Script to extract metadata from microscopy image files."""

from lib.preprocess.preprocess import extract_metadata, get_data_config

# Get data configuration from rule name
rule_name = snakemake.rule
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Extract metadata parameters
cycle = getattr(snakemake.params, "cycle", None)
round_param = getattr(snakemake.params, "round", None)

# Get the sample files and metadata file from the structured input
sample_files = getattr(snakemake.input, "samples", [])
metadata_files = getattr(snakemake.input, "metadata", [])

# Determine what we're extracting from
if metadata_files:
    # Extract from metadata CSV files
    metadata_file_path = metadata_files[0]
    file_input = metadata_file_path  # Pass the CSV file as the main input
    print(f"Using metadata file: {metadata_file_path}")
elif sample_files:
    # Extract from image files
    file_input = sample_files
    metadata_file_path = None
    print("No metadata file - extracting from image headers")
else:
    raise ValueError("No input files provided - need either samples or metadata")

# Extract metadata using main function
metadata_df = extract_metadata(
    file_input,
    plate=snakemake.params.plate,
    well=snakemake.params.well,
    tile=snakemake.params.tile,
    cycle=cycle,
    round=round_param,
    data_format=data_config["data_format"],
    data_organization=data_config["image_data_organization"],
    metadata_file_path=metadata_file_path,
    verbose=False,
)

# Save the extracted metadata
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
