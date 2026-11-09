"""Script to extract metadata from microscopy image files."""

from itertools import groupby

import pandas as pd

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
    # Extract from image files (e.g. ND2, IMS)
    file_input = sample_files
    metadata_file_path = None
    print("No metadata file - extracting from image headers")
else:
    raise ValueError("No input files provided - need either samples or metadata")

# One job covers a well: tile-organized files are read tile by tile, as the sample table assigns them
if metadata_file_path is None and data_config["image_data_organization"] == "tile":
    samples_fp = snakemake.config["preprocess"][f"{image_type}_samples_fp"]
    samples = pd.read_csv(samples_fp, sep="\t")
    tile_of = dict(zip(samples["sample_fp"].astype(str), samples["tile"]))
    tile_files = [
        (tile, list(files))
        for tile, files in groupby(file_input, key=lambda fp: tile_of[str(fp)])
    ]
else:
    tile_files = [(snakemake.params.tile, file_input)]

# Extract metadata using main function
metadata_df = pd.concat(
    [
        extract_metadata(
            files,
            plate=snakemake.params.plate,
            well=snakemake.params.well,
            tile=tile,
            cycle=cycle,
            round=round_param,
            data_format=data_config["data_format"],
            data_organization=data_config["image_data_organization"],
            metadata_file_path=metadata_file_path,
            verbose=False,
        )
        for tile, files in tile_files
    ],
    ignore_index=True,
)

# Save the extracted metadata
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
