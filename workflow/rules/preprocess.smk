import glob
from pathlib import Path

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]

# Load the sample TSV files with pandas
sbs_samples_df = pd.read_csv(SBS_SAMPLES_FP, sep="\t")
phenotype_samples_df = pd.read_csv(PHENOTYPE_SAMPLES_FP, sep="\t")

# Print the shape of each DataFrame
print(f"SBS samples shape: {sbs_samples_df.shape}")
print(f"Phenotype samples shape: {phenotype_samples_df.shape}")

# File patterns for SBS and PH images with placeholders (find all tiles to compile metadata)
SBS_INPUT_PATTERN_METADATA = "input/sbs/*C{cycle}_Wells-{well}_Points-*__Channel*.nd2"

# Extract metadata for SBS images
rule extract_metadata_sbs:
    input:
        lambda wildcards: sbs_samples_df[
            (sbs_samples_df["well"] == wildcards.well) & 
            (sbs_samples_df["cycle"] == wildcards.cycle)
        ]["sample_fp"].tolist(),
    output:
        PREPROCESS_FP / "10X_c{cycle}-SBS-{cycle}_{well}.metadata.pkl"
    script:
        "../scripts/preprocess/extract_metadata_tile.py"