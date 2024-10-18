from pathlib import Path

from lib.preprocess.file_utils import get_sample_fps

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]

# Load the sample TSV files with pandas
sbs_samples_df = pd.read_csv(SBS_SAMPLES_FP, sep="\t")
phenotype_samples_df = pd.read_csv(PHENOTYPE_SAMPLES_FP, sep="\t")

# Print the shape of each DataFrame
print(f"SBS samples shape: {sbs_samples_df.shape}")
print(f"Phenotype samples shape: {phenotype_samples_df.shape}")

# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        lambda wildcards: get_sample_fps(
            sbs_samples_df,
            well=wildcards.well,
            cycle=wildcards.cycle
        )
    output:
        PREPROCESS_FP / "10X_c{cycle}-SBS-{cycle}_{well}.metadata.tsv"
    script:
        "../scripts/preprocess/extract_metadata_tile.py"