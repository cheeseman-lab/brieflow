from tifffile import imread, imwrite

from lib.phenotype.align_channels import align_phenotype_channels
from lib.shared.align import apply_custom_offsets

# Load image data
image_data = imread(snakemake.input[0])

# Get the alignment config
align_config = snakemake.params.config
print("Alignment config:", align_config)

# Standard alignment process
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
                remove_channel=step["remove_channel"],
                upsample_factor=step.get(
                    "upsample_factor", align_config.get("upsample_factor", 2)
                ),
                window=step.get("window", align_config.get("window", 2)),
            )
    else:
        # Handle single-step alignment
        print("Performing single-step alignment...")
        aligned_data = align_phenotype_channels(
            image_data,
            target=align_config["target"],
            source=align_config["source"],
            riders=align_config.get("riders", []),
            remove_channel=align_config["remove_channel"],
            upsample_factor=align_config.get("upsample_factor", 2),
            window=align_config.get("window", 2),
        )
else:
    print("Skipping alignment...")
    aligned_data = image_data

# Custom alignment process (applies after standard alignment)
if align_config.get("custom_channel_offsets"):
    print("Applying custom channel offsets...")
    print(f"Custom offsets: {align_config['custom_channel_offsets']}")

    # Apply custom offsets using the dict from config
    aligned_data = apply_custom_offsets(
        aligned_data,
        offsets_dict=align_config["custom_channel_offsets"],
    )

# Save the aligned/unaligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)
