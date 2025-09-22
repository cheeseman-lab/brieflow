"""Snakemake script for stitching well tiles into complete well images."""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import (
    assemble_aligned_tiff_well,
    extract_cell_positions_from_stitched_mask,
    assemble_stitched_masks,
)
from lib.merge.eval_stitch import create_tile_arrangement_qc_plot

data_type = snakemake.params.data_type
plate = snakemake.params.plate
well = snakemake.params.well
overlap_fraction = snakemake.params.overlap_fraction

print(f"Stitching {data_type} - plate {plate}, well {well}")

# Load metadata
print("Loading metadata...")
if data_type == "phenotype":
    metadata = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_metadata))
else:  # sbs
    metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))

# Apply SBS-specific filtering if needed
if data_type == "sbs":
    sbs_filters = {"cycle": snakemake.config["merge"]["sbs_metadata_cycle"]}
    for filter_key, filter_value in sbs_filters.items():
        metadata = metadata[metadata[filter_key] == filter_value]
    print(f"Applied SBS filter: {len(metadata)} entries remaining")

# Load stitching configuration
print("Loading stitch configuration...")
if data_type == "phenotype":
    stitch_config_path = snakemake.input.phenotype_stitch_config
else:  # sbs
    stitch_config_path = snakemake.input.sbs_stitch_config

with open(stitch_config_path, "r") as f:
    stitch_config = yaml.safe_load(f)

print(f"Starting {data_type} stitching for plate {plate}, well {well}")

# Filter metadata to specific plate and well
well_metadata = metadata[(metadata["plate"] == int(plate)) & (metadata["well"] == well)]

if len(well_metadata) == 0:
    raise ValueError(f"No {data_type} tiles found for plate {plate}, well {well}")

print(f"Found {len(well_metadata)} {data_type} tiles")

# Build tile file mappings from Snakemake inputs
print("Building tile file mappings from inputs...")

# Get tile files from Snakemake input
if data_type == "phenotype":
    tile_files_list = snakemake.input.phenotype_tiles
    mask_files_list = snakemake.input.phenotype_masks
else:
    tile_files_list = snakemake.input.sbs_tiles
    mask_files_list = snakemake.input.sbs_masks

# Create tile_id -> file_path mappings
tile_files = {}
mask_files = {}

for tile_file in tile_files_list:
    # Extract tile_id from filename pattern: P-{plate}_W-{well}_T-{tile}__aligned.tiff
    tile_file_path = Path(tile_file)
    filename = tile_file_path.stem  # Remove .tiff extension
    
    # Parse tile ID from filename
    parts = filename.split('_')
    for part in parts:
        if part.startswith('T-'):
            tile_id = int(part.split('-')[1].split('__')[0])
            tile_files[tile_id] = str(tile_file)
            break

for mask_file in mask_files_list:
    # Extract tile_id from filename pattern: P-{plate}_W-{well}_T-{tile}__nuclei.tiff
    mask_file_path = Path(mask_file)
    filename = mask_file_path.stem
    
    # Parse tile ID from filename
    parts = filename.split('_')
    for part in parts:
        if part.startswith('T-'):
            tile_id = int(part.split('-')[1].split('__')[0])
            mask_files[tile_id] = str(mask_file)
            break

print(f"Found {len(tile_files)} tile files and {len(mask_files)} mask files")

# Validate stitch configuration
if "total_translation" not in stitch_config:
    raise ValueError("Stitch config missing 'total_translation' data")

shifts = stitch_config["total_translation"]
if len(shifts) == 0:
    raise ValueError("No tile shifts found in stitch config")

print(f"Using {len(shifts)} tile shifts from stitch config")

# Determine tile size
tile_size = (2400, 2400) if data_type == "phenotype" else (1200, 1200)
print(f"Using tile size: {tile_size}")

# Check if stitched image creation is enabled
create_stitched_image = getattr(snakemake.params, "stitched_image", True)

# Assemble stitched image
if create_stitched_image:
    print("Assembling stitched image...")
    try:
        stitched_image = assemble_aligned_tiff_well(
            tile_files=tile_files,
            shifts=shifts,
            well=well,
            tile_size=tile_size,
            flipud=snakemake.params.flipud,
            fliplr=snakemake.params.fliplr,
            rot90=snakemake.params.rot90,
            channel=0,
        )
        print(f"Stitched image created: {stitched_image.shape}")
    except Exception as e:
        raise RuntimeError(f"Image stitching failed: {e}")
else:
    print("Skipping stitched image creation")
    stitched_image = np.array([[0]], dtype=np.uint16)

# Initialize cell positions
cell_positions = pd.DataFrame(columns=["well", "cell", "i", "j", "area", "data_type"])

# Assemble stitched masks
masks_exist = len(mask_files) > 0

if masks_exist:
    print("Assembling stitched masks...")
    try:
        stitched_mask, cell_id_mapping = assemble_stitched_masks(
            mask_files=mask_files,
            shifts=shifts,
            well=well,
            tile_size=tile_size,
            flipud=snakemake.params.flipud,
            fliplr=snakemake.params.fliplr,
            rot90=snakemake.params.rot90,
            return_cell_mapping=True,
        )
        print(
            f"Stitched mask created: {stitched_mask.shape}, max label: {stitched_mask.max()}"
        )

        # Extract cell positions
        if stitched_mask.max() > 0:
            print("Extracting cell positions...")

            cell_positions = extract_cell_positions_from_stitched_mask(
                stitched_mask=stitched_mask,
                well=well,
                plate=plate,
                tile_metadata=well_metadata,
                shifts=shifts,
                tile_size=tile_size,
                cell_id_mapping=cell_id_mapping,
                data_type=data_type,
            )
            print(f"Extracted {len(cell_positions)} cell positions")
        else:
            print("No cells found in stitched mask")

    except Exception as e:
        print(f"Mask stitching failed: {e}")
        mask_shape = stitched_image.shape if create_stitched_image else (1, 1)
        stitched_mask = np.zeros(mask_shape, dtype=np.uint16)
        print("Created empty mask, continuing with image only")
else:
    mask_shape = stitched_image.shape if create_stitched_image else (1, 1)
    stitched_mask = np.zeros(mask_shape, dtype=np.uint16)
    print("Created empty mask (no mask files available)")

# Save outputs - positions [0], qc [1], then conditional image [2], mask [3]
print("Saving outputs...")

# Always save cell positions (index 0)
if hasattr(snakemake.output, "phenotype_cell_positions"):
    output_positions = snakemake.output.phenotype_cell_positions
elif hasattr(snakemake.output, "sbs_cell_positions"):
    output_positions = snakemake.output.sbs_cell_positions
else:
    output_positions = snakemake.output[0]

cell_positions.to_parquet(output_positions)
print(f"Cell positions saved: {output_positions} ({len(cell_positions)} cells)")

# Always save QC plot (index 1)
if hasattr(snakemake.output, "phenotype_qc_plot"):
    output_qc = snakemake.output.phenotype_qc_plot
elif hasattr(snakemake.output, "sbs_qc_plot"):
    output_qc = snakemake.output.sbs_qc_plot
else:
    output_qc = snakemake.output[1]

if len(cell_positions) > 0:
    print("Creating QC plot...")
    create_tile_arrangement_qc_plot(
        cell_positions_df=cell_positions,
        output_path=output_qc,
        data_type=data_type,
        well=well,
        plate=plate,
    )
    print(f"QC plot saved: {output_qc}")
else:
    print("No cell positions available, creating empty QC plot...")
    # Create a minimal placeholder plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(
        0.5,
        0.5,
        f"No cells found for {data_type} well {well}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.set_title(f"{data_type.title()} Well {well} - No Cells Detected")
    plt.tight_layout()
    plt.savefig(output_qc, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Empty QC plot saved: {output_qc}")

# Conditionally save stitched image and mask when enabled
if create_stitched_image:
    # Save stitched image (index 2)
    if hasattr(snakemake.output, "phenotype_stitched_image"):
        output_image = snakemake.output.phenotype_stitched_image
    elif hasattr(snakemake.output, "sbs_stitched_image"):
        output_image = snakemake.output.sbs_stitched_image
    elif len(snakemake.output) > 2:
        output_image = snakemake.output[2]
    else:
        output_image = None

    if output_image:
        np.save(output_image, stitched_image)
        print(f"Stitched image saved: {output_image}")

    # Save stitched mask (index 3)
    if hasattr(snakemake.output, "phenotype_stitched_mask"):
        output_mask = snakemake.output.phenotype_stitched_mask
    elif hasattr(snakemake.output, "sbs_stitched_mask"):
        output_mask = snakemake.output.sbs_stitched_mask
    elif len(snakemake.output) > 3:
        output_mask = snakemake.output[3]
    else:
        output_mask = None

    if output_mask:
        np.save(output_mask, stitched_mask)
        print(f"Stitched mask saved: {output_mask}")
else:
    print("Skipping stitched image and mask saving (stitched_image=False)")
    # Create minimal empty files to satisfy Snakemake output requirements

    # Create empty image file (index 2)
    if hasattr(snakemake.output, "phenotype_stitched_image"):
        output_image = snakemake.output.phenotype_stitched_image
    elif hasattr(snakemake.output, "sbs_stitched_image"):
        output_image = snakemake.output.sbs_stitched_image
    elif len(snakemake.output) > 2:
        output_image = snakemake.output[2]
    else:
        output_image = None

    if output_image:
        placeholder_image = np.array([[0]], dtype=np.uint16)
        np.save(output_image, placeholder_image)
        print(f"Empty placeholder image saved: {output_image}")

    # Create empty mask file (index 3)
    if hasattr(snakemake.output, "phenotype_stitched_mask"):
        output_mask = snakemake.output.phenotype_stitched_mask
    elif hasattr(snakemake.output, "sbs_stitched_mask"):
        output_mask = snakemake.output.sbs_stitched_mask
    elif len(snakemake.output) > 3:
        output_mask = snakemake.output[3]
    else:
        output_mask = None

    if output_mask:
        placeholder_mask = np.array([[0]], dtype=np.uint16)
        np.save(output_mask, placeholder_mask)
        print(f"Empty placeholder mask saved: {output_mask}")

print(f"{data_type} stitching completed successfully")