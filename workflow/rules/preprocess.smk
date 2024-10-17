from pathlib import Path

PREPROCESS_FP = ROOT_FP / config["preprocess"]["suffix"]

# Load the sample TSV files with pandas
sbs_samples_df = pd.read_csv(SBS_SAMPLES_FP, sep="\t")
phenotype_samples_df = pd.read_csv(PHENOTYPE_SAMPLES_FP, sep="\t")

# Print the shape of each DataFrame
print(f"SBS samples shape: {sbs_samples_df.shape}")
print(f"Phenotype samples shape: {phenotype_samples_df.shape}")

# Create a test rule to transpose and save the DataFrame
rule save_transposed_df:
    output:
        PREPROCESS_FP / "transposed_df.tsv"
    run:
        print("Transposing the DataFrame...")
        # Transpose the DataFrame
        transposed_df = sbs_samples_df.transpose()
        # Save the transposed DataFrame to the output file
        transposed_df.to_csv(output[0], sep="\t")
        print(f"Transposed DataFrame saved to {output[0]}")