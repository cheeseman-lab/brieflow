from pathlib import Path

from lib.shared.file_utils import get_sample_fps

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]

# Load the sample TSV files with pandas
sbs_samples_df = pd.read_csv(SBS_SAMPLES_FP, sep="\t")
phenotype_samples_df = pd.read_csv(PHENOTYPE_SAMPLES_FP, sep="\t")

# Print the shape of each DataFrame
print(f"SBS samples shape: {sbs_samples_df.shape}")
print(f"Phenotype samples shape: {phenotype_samples_df.shape}")


# Extract metadata for SBS images
rule extract_metadata_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df, well=wildcards.well, cycle=wildcards.cycle
        ),
    output:
        PREPROCESS_FP / "metadata" / "10X_c{cycle}-SBS-{cycle}_{well}.metadata.tsv",
    script:
        "../scripts/preprocess/extract_metadata_tile.py"


# Extract metadata for phenotype images
rule extract_metadata_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(phenotype_samples_df, well=wildcards.well),
    output:
        PREPROCESS_FP / "metadata" / "20X_{well}.metadata.tsv",
    script:
        "../scripts/preprocess/extract_metadata_tile.py"


# Convert SBS ND2 files to TIFF
rule convert_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            well=wildcards.well,
            cycle=wildcards.cycle,
            tile=wildcards.tile,
        ),
    output:
        PREPROCESS_FP
        / "sbs_tifs"
        / "10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif",
    params:
        channel_order_flip=True,
    script:
        "../scripts/preprocess/nd2_to_tif.py"


# Convert phenotype ND2 files to TIFF
rule convert_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: get_sample_fps(
            phenotype_samples_df, well=wildcards.well, tile=wildcards.tile
        ),
    output:
        PREPROCESS_FP / "phenotype_tifs" / "20X_{well}_Tile-{tile}.phenotype.tif",
    params:
        channel_order_flip=True,
    script:
        "../scripts/preprocess/nd2_to_tif.py"


# Calculate illumination correction function for sbs files
rule calculate_ic_sbs:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: expand(
            PREPROCESS_FP
            / "sbs_tifs"
            / "10X_c{cycle}-SBS-{cycle}_{well}_Tile-{tile}.sbs.tif",
            well=wildcards.well,
            tile=SBS_TILES,
            cycle=wildcards.cycle,
        ),
    output:
        PREPROCESS_FP / "ic_fields" / "10X_c{cycle}-SBS-{cycle}_{well}.sbs.ic_field.tif",
    params:
        threading=True,
    script:
        "../scripts/preprocess/calculate_ic_field.py"


# Calculate illumination correction for phenotype files
rule calculate_ic_phenotype:
    conda:
        "../envs/preprocess.yml"
    input:
        lambda wildcards: expand(
            PREPROCESS_FP / "phenotype_tifs" / "20X_{well}_Tile-{tile}.phenotype.tif",
            well=wildcards.well,
            tile=PHENOTYPE_TILES,
        ),
    output:
        PREPROCESS_FP / "ic_fields" / "20X_{well}.phenotype.ic_field.tif",
    params:
        threading=True,
    script:
        "../scripts/preprocess/calculate_ic_field.py"
