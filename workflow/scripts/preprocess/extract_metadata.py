"""Script to extract metadata from microscopy image files."""

import pandas as pd
from pathlib import Path
from lib.preprocess.preprocess import extract_metadata, get_data_config
from lib.preprocess.file_utils import get_sample_fps

# Get data configuration from rule name
rule_name = (
    snakemake.rule
)  # Will be "extract_metadata_sbs" or "extract_metadata_phenotype"
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(
    image_type, {"preprocess": snakemake.config.get("preprocess", {})}
)

# Extract metadata parameters
cycle = getattr(snakemake.params, "cycle", None)
round_param = getattr(snakemake.params, "round", None)

# Determine metadata file path if external metadata is configured
metadata_file_path = None
metadata_samples_df_fp = data_config.get("metadata_samples_df_fp")

if metadata_samples_df_fp and Path(metadata_samples_df_fp).exists():
    # Load the metadata samples dataframe
    metadata_samples_df = pd.read_csv(metadata_samples_df_fp, sep="\t")

    # Build filter criteria based on available parameters
    filter_criteria = {
        "plate": snakemake.params.plate,
    }

    # Add cycle or round if available
    if image_type == "sbs" and cycle is not None:
        filter_criteria["cycle"] = cycle
    elif image_type == "phenotype" and round_param is not None:
        filter_criteria["round"] = round_param

    # Find matching metadata file
    try:
        metadata_file_path = get_sample_fps(
            metadata_samples_df, **filter_criteria, verbose=False
        )
        print(f"Using metadata file: {metadata_file_path}")
    except Exception as e:
        print(f"Could not find metadata file: {e}")
        metadata_file_path = None

# Extract metadata using main function
metadata_df = extract_metadata(
    snakemake.input[0],
    plate=snakemake.params.plate,
    well=snakemake.params.well,
    tile=snakemake.params.tile,
    cycle=cycle,
    round=round_param,
    data_format=data_config["data_format"],
    data_organization=data_config["data_organization"],
    metadata_file_path=metadata_file_path,
    verbose=False,
)

# Save the extracted metadata
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
