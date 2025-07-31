from lib.preprocess.preprocess import extract_metadata_unified, get_data_config

# Get data configuration from rule name
rule_name = snakemake.rule  # Will be "extract_metadata_sbs" or "extract_metadata_phenotype"
image_type = "sbs" if "sbs" in rule_name else "phenotype"

data_config = get_data_config(image_type, {"preprocess": snakemake.config.get("preprocess", {})})

# Extract metadata parameters
cycle = getattr(snakemake.params, "cycle", None)

# Extract metadata using unified function
metadata_df = extract_metadata_unified(
    snakemake.input[0],
    plate=snakemake.params.plate,
    well=snakemake.params.well,
    tile=snakemake.params.tile,
    cycle=cycle,
    data_format=data_config["data_format"],
    data_organization=data_config["data_organization"],
    verbose=False
)

# Save the extracted metadata
metadata_df.to_csv(snakemake.output[0], index=False, sep="\t")
