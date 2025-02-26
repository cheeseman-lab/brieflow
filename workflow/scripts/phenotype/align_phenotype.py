from tifffile import imread, imwrite

from lib.phenotype.align_channels import align_phenotype_channels

# Load image data
image_data = imread(snakemake.input[0])

# Get the alignment config
align_config = snakemake.params.config
print("Alignment config:", align_config)

if align_config["align"]:
    print("Aligning channels...")
    
    if align_config["multi_step"]:
        # Handle multi-step alignment
        print(f"Performing {len(align_config['steps'])}-step alignment...")
        aligned_data = image_data
        
        for i, step in enumerate(align_config["steps"], 1):
            print(f"Step {i}: Aligning channels...")
            print(f"Step parameters: {step}")
            aligned_data = align_phenotype_channels(
                aligned_data,
                target=step["target"],
                source=step["source"],
                riders=step.get("riders", []),
                remove_channel=step["remove_channel"]
            )
    else:
        # Handle single-step alignment
        print("Performing single-step alignment...")
        aligned_data = align_phenotype_channels(
            image_data,
            target=align_config["target"],
            source=align_config["source"],
            riders=align_config.get("riders", []),
            remove_channel=align_config["remove_channel"]
        )
else:
    print("Skipping alignment...")
    aligned_data = image_data

# Save the aligned/unaligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
