import pandas as pd
from tifffile import imread, imwrite

from lib.phenotype.align_channels import align_phenotype_channels
from lib.shared.align import apply_custom_offsets

# Load image data
image_data = imread(snakemake.input[0])

# Get the alignment config
align_config = snakemake.params.config
print("Alignment config:", align_config)

# Start with original image data
aligned_data = image_data

# Dictionary to collect all alignment metrics
all_metrics = {}

# STEP 1: Apply custom offsets FIRST (if they exist)
if align_config.get("custom_channel_offsets"):
    print("STEP 1: Applying custom channel offsets...")
    print(f"Custom offsets: {align_config.get('custom_channel_offsets')}")

    aligned_data = apply_custom_offsets(
        aligned_data,
        offsets_dict=align_config.get("custom_channel_offsets"),
    )
else:
    print("STEP 1: No custom offsets to apply")

# STEP 2: Apply automatic alignment SECOND (if enabled)
if align_config["align"]:
    print("STEP 2: Applying automatic alignment...")

    if align_config["multi_step"]:
        # Handle multi-step alignment
        print(f"Performing {len(align_config['steps'])}-step alignment...")

        for i, step in enumerate(align_config["steps"], 1):
            print(f"  Step {i}: Aligning channels...")
            print(f"  Step parameters: {step}")
            aligned_data, metrics = align_phenotype_channels(
                aligned_data,
                target=step["target"],
                source=step["source"],
                riders=step.get("riders", []),
                remove_channel=step["remove_channel"],
                upsample_factor=step.get(
                    "upsample_factor", align_config.get("upsample_factor", 2)
                ),
                window=step.get("window", align_config.get("window", 2)),
                return_metrics=True,
            )
            # Add step-suffixed metrics
            all_metrics[f"align_confidence_step{i}"] = metrics["align_confidence"]
            all_metrics[f"offset_y_step{i}"] = metrics["offset"][0]
            all_metrics[f"offset_x_step{i}"] = metrics["offset"][1]
            print(
                f"  Step {i} metrics: align_confidence={metrics['align_confidence']:.3f}"
            )
    else:
        # Handle single-step alignment
        print("Performing single-step alignment...")
        aligned_data, metrics = align_phenotype_channels(
            aligned_data,
            target=align_config["target"],
            source=align_config["source"],
            riders=align_config.get("riders", []),
            remove_channel=align_config["remove_channel"],
            upsample_factor=align_config.get("upsample_factor", 2),
            window=align_config.get("window", 2),
            return_metrics=True,
        )
        # Add metrics without step suffix for single-step
        all_metrics["align_confidence"] = metrics["align_confidence"]
        all_metrics["offset_y"] = metrics["offset"][0]
        all_metrics["offset_x"] = metrics["offset"][1]
        print(f"  Metrics: align_confidence={metrics['align_confidence']:.3f}")
else:
    print("STEP 2: Skipping automatic alignment")
    # No alignment - write placeholder metrics
    all_metrics["align_confidence"] = None
    all_metrics["offset_y"] = 0
    all_metrics["offset_x"] = 0

# Save the aligned/unaligned data as a .tiff file
imwrite(snakemake.output[0], aligned_data)

# Save alignment metrics to TSV (one row per tile)
metrics_df = pd.DataFrame(
    [
        {
            "plate": snakemake.wildcards.plate,
            "well": snakemake.wildcards.well,
            "tile": snakemake.wildcards.tile,
            **all_metrics,
        }
    ]
)
metrics_df.to_csv(snakemake.output[1], index=False, sep="\t")
print(f"Alignment metrics saved to {snakemake.output[1]}")
