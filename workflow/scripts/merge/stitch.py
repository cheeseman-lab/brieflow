"""Stitches tiles into wells and extracts cell positions.

Applies transformations calculated in estimate stitch, assigns global cell
coordinates and ID's. Optionally, generates stitched masks and images for QC.
"""

import pandas as pd
import numpy as np
import yaml

from lib.shared.file_utils import validate_dtypes, files_to_tile_mapping
from lib.merge.stitch import (
    assemble_aligned_tiff_well,
    extract_cell_positions_from_stitched_mask,
    assemble_stitched_masks,
)
from lib.merge.eval_stitch import create_tile_arrangement_qc_plot, create_empty_qc_plot

# Extract parameters
data_type = snakemake.params.data_type
plate = snakemake.params.plate
well = snakemake.params.well
create_stitched_image = getattr(snakemake.params, "stitched_image", True)

print(f"Stitching {data_type} - plate {plate}, well {well}")

# Load metadata
print("Loading metadata...")
metadata = validate_dtypes(
    pd.read_parquet(
        snakemake.input.phenotype_metadata
        if data_type == "phenotype"
        else snakemake.input.sbs_metadata
    )
)

# Build and apply metadata filters based on data type
filters = {}
if data_type == "sbs":
    if getattr(snakemake.params, "sbs_metadata_cycle", None) is not None:
        filters["cycle"] = snakemake.params.sbs_metadata_cycle
    if getattr(snakemake.params, "sbs_metadata_channel", None) is not None:
        filters["channel"] = snakemake.params.sbs_metadata_channel
elif data_type == "phenotype":
    if getattr(snakemake.params, "ph_metadata_channel", None) is not None:
        filters["channel"] = snakemake.params.ph_metadata_channel

for filter_key, filter_value in filters.items():
    metadata = metadata[metadata[filter_key] == filter_value]
    print(
        f"Applied {data_type} filter {filter_key}={filter_value}: {len(metadata)} entries remaining"
    )

# If no filters were applied, deduplicate to handle multiple channels per tile
if not filters:
    metadata = metadata.drop_duplicates(subset=["plate", "well", "tile"])
    print(f"Deduplicated metadata: {len(metadata)} unique tiles")

# Load stitch configuration
print("Loading stitch configuration...")
stitch_config_path = (
    snakemake.input.phenotype_stitch_config
    if data_type == "phenotype"
    else snakemake.input.sbs_stitch_config
)

with open(stitch_config_path, "r") as f:
    stitch_config = yaml.safe_load(f)

if "total_translation" not in stitch_config:
    raise ValueError("Stitch config missing 'total_translation' data")

shifts = stitch_config["total_translation"]
print(f"Using {len(shifts)} tile shifts from stitch config")

# Parse tile files
print("Building tile file mappings from inputs...")
if data_type == "phenotype":
    tile_files = files_to_tile_mapping(snakemake.input.phenotype_tiles)
    mask_files = files_to_tile_mapping(snakemake.input.phenotype_masks)
else:
    tile_files = files_to_tile_mapping(snakemake.input.sbs_tiles)
    mask_files = files_to_tile_mapping(snakemake.input.sbs_masks)

print(f"Found {len(tile_files)} tile files and {len(mask_files)} mask files")

# Assemble stitched image
if create_stitched_image:
    print("Assembling stitched image...")
    try:
        stitched_image = assemble_aligned_tiff_well(
            tile_files=tile_files,
            shifts=shifts,
            well=well,
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

# Assemble stitched masks and extract positions
if mask_files:
    print("Assembling stitched masks...")
    try:
        stitched_mask, cell_id_mapping = assemble_stitched_masks(
            mask_files=mask_files,
            shifts=shifts,
            well=well,
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
                tile_metadata=metadata,
                shifts=shifts,
                cell_id_mapping=cell_id_mapping,
                data_type=data_type,
            )
            print(f"Extracted {len(cell_positions)} cell positions")
        else:
            print("No cells found in stitched mask")
            cell_positions = pd.DataFrame()
    except Exception as e:
        print(f"Mask stitching failed: {e}")
        stitched_mask = np.zeros(stitched_image.shape, dtype=np.uint16)
        cell_positions = pd.DataFrame()
else:
    stitched_mask = np.zeros(stitched_image.shape, dtype=np.uint16)
    cell_positions = pd.DataFrame()
    print("No mask files available, created empty mask")

# Save outputs
print("Saving outputs...")

# Save cell positions
output_positions = getattr(
    snakemake.output, f"{data_type}_cell_positions", snakemake.output[0]
)
cell_positions.to_parquet(output_positions)
print(f"Cell positions saved: {output_positions} ({len(cell_positions)} cells)")

# Save QC plot
output_qc = getattr(snakemake.output, f"{data_type}_qc_plot", snakemake.output[1])

if len(cell_positions) > 0:
    print("Creating QC plot...")
    create_tile_arrangement_qc_plot(cell_positions, output_qc, data_type, well, plate)
else:
    print("Creating empty QC plot (no cells found)...")
    create_empty_qc_plot(output_qc, data_type, well)

print(f"QC plot saved: {output_qc}")

# Save stitched image and mask
output_image = getattr(
    snakemake.output, f"{data_type}_stitched_image", snakemake.output[2]
)
output_mask = getattr(
    snakemake.output, f"{data_type}_stitched_mask", snakemake.output[3]
)

if create_stitched_image:
    np.save(output_image, stitched_image)
    np.save(output_mask, stitched_mask)
    print(f"Stitched image saved: {output_image}")
    print(f"Stitched mask saved: {output_mask}")
else:
    # Create empty placeholder files to satisfy Snakemake
    placeholder = np.array([[0]], dtype=np.uint16)
    np.save(output_image, placeholder)
    np.save(output_mask, placeholder)
    print("Empty placeholder files saved (stitched_image=False)")

print(f"{data_type} stitching completed successfully")
