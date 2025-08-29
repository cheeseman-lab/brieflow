"""Snakemake script for stitching well tiles into complete well images.

This script handles the assembly of individual microscopy tiles into complete
well images and masks, with preserved cell ID mapping for downstream analysis.
It supports both phenotype and SBS data types with configurable transformations.
"""

import pandas as pd
import numpy as np
import yaml
import gc
import psutil
import os
from pathlib import Path
from skimage import io

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import (
    assemble_aligned_tiff_well,
    extract_cell_positions_from_stitched_mask,
    create_tile_arrangement_qc_plot,
    assemble_stitched_masks_simple,
)


def print_progress(message):
    """Print progress messages with consistent formatting."""
    print(f"🔄 {message}")


def print_success(message):
    """Print success messages with consistent formatting."""
    print(f"✅ {message}")


def print_error(message):
    """Print error messages with consistent formatting."""
    print(f"❌ {message}")


def print_warning(message):
    """Print warning messages with consistent formatting."""
    print(f"⚠️  {message}")


def check_mask_files_exist(metadata_df, well, data_type):
    """Check if nuclei mask files exist for the specified well.

    This function validates that the required segmentation mask files are
    available before attempting to process them.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata containing tile information
    well : str
        Well identifier to check
    data_type : str
        Data type ('phenotype' or 'sbs')

    Returns:
    -------
    Tuple[bool, List[str]]
        (files_exist, list_of_existing_files)
    """
    well_metadata = metadata_df[metadata_df["well"] == well]

    if len(well_metadata) == 0:
        return False, []

    existing_masks = []
    # Check first 3 tiles as a representative sample
    for _, row in well_metadata.head(3).iterrows():
        plate = row["plate"]
        tile_id = row["tile"]
        mask_path = (
            f"analysis_root/{data_type}/images/"
            f"P-{plate}_W-{well}_T-{tile_id}__nuclei.tiff"
        )
        if Path(mask_path).exists():
            existing_masks.append(mask_path)

    masks_exist = len(existing_masks) > 0
    return masks_exist, existing_masks


def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()


def process_stitching(metadata_df, stitch_config, plate, well, data_type, params):
    """Main stitching processing function.

    This function handles the complete stitching workflow including image
    assembly, mask processing, and cell position extraction. It implements
    robust error handling to process what it can while failing fast on
    critical errors.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata for all tiles
    stitch_config : dict
        Stitching configuration with translation shifts
    plate : str
        Plate identifier
    well : str
        Well identifier
    data_type : str
        Data type ('phenotype' or 'sbs')
    params : object
        Snakemake parameters object with processing options

    Returns:
    -------
    dict
        Dictionary containing stitched_image, stitched_mask, and cell_positions

    Raises:
    ------
    ValueError
        If no tiles found or critical configuration missing
    RuntimeError
        If image stitching fails
    """
    print_progress(
        f"Starting {data_type.upper()} stitching for Plate {plate}, Well {well}"
    )

    # Filter metadata to specific plate and well
    well_metadata = metadata_df[
        (metadata_df["plate"] == int(plate)) & (metadata_df["well"] == well)
    ]

    if len(well_metadata) == 0:
        raise ValueError(f"No {data_type} tiles found for plate {plate}, well {well}")

    print_progress(f"Found {len(well_metadata)} {data_type} tiles")

    # Check mask file availability
    masks_exist, existing_masks = check_mask_files_exist(well_metadata, well, data_type)
    if masks_exist:
        print_progress(f"Nuclei masks available: {len(existing_masks)} found")
    else:
        print_warning(f"No nuclei masks found - will create empty mask")

    # Validate stitch configuration
    if "total_translation" not in stitch_config:
        raise ValueError("Stitch config missing 'total_translation' data")

    shifts = stitch_config["total_translation"]
    if len(shifts) == 0:
        raise ValueError("No tile shifts found in stitch config")

    print_progress(f"Using {len(shifts)} tile shifts from stitch config")

    # Step 1: Assemble stitched image
    print_progress("Assembling stitched image...")
    try:
        stitched_image = assemble_aligned_tiff_well(
            metadata_df=well_metadata,
            shifts=shifts,
            well=well,
            data_type=data_type,
            flipud=params.flipud,
            fliplr=params.fliplr,
            rot90=params.rot90,
        )
        print_success(f"Stitched image created: {stitched_image.shape}")
    except Exception as e:
        raise RuntimeError(f"Image stitching failed: {e}")

    # Step 2: Assemble stitched masks (if available)
    cell_positions = pd.DataFrame(
        columns=["well", "cell", "i", "j", "area", "data_type"]
    )

    if masks_exist:
        print_progress("Assembling stitched masks...")
        try:
            stitched_mask, cell_id_mapping = assemble_stitched_masks_simple(
                metadata_df=well_metadata,
                shifts=shifts,
                well=well,
                data_type=data_type,
                flipud=params.flipud,
                fliplr=params.fliplr,
                rot90=params.rot90,
                return_cell_mapping=True,
            )
            print_success(
                f"Stitched mask created: {stitched_mask.shape}, "
                f"{stitched_mask.max()} max label"
            )

            # Step 3: Extract cell positions
            if stitched_mask.max() > 0:
                print_progress("Extracting cell positions...")
                tile_size = (2400, 2400) if data_type == "phenotype" else (1200, 1200)

                cell_positions = extract_cell_positions_from_stitched_mask(
                    stitched_mask=stitched_mask,
                    well=well,
                    plate=plate,
                    data_type=data_type,
                    metadata_df=well_metadata,
                    shifts=shifts,
                    tile_size=tile_size,
                    cell_id_mapping=cell_id_mapping,
                )
                print_success(f"Extracted {len(cell_positions)} cell positions")
            else:
                print_warning("No cells found in stitched mask")

        except Exception as e:
            print_error(f"Mask stitching failed: {e}")
            # Create empty mask but continue with image processing
            stitched_mask = np.zeros(stitched_image.shape, dtype=np.uint16)
            print_warning("Created empty mask, continuing with image only")
    else:
        # Create empty mask when no mask files are available
        stitched_mask = np.zeros(stitched_image.shape, dtype=np.uint16)
        print_progress("Created empty mask (no mask files available)")

    cleanup_memory()

    return {
        "stitched_image": stitched_image,
        "stitched_mask": stitched_mask,
        "cell_positions": cell_positions,
    }


def create_qc_plot(cell_positions, plate, well, data_type, output_dir):
    """Create quality control plot if cell positions are available.

    Parameters
    ----------
    cell_positions : pd.DataFrame
        Cell positions dataframe
    plate : str
        Plate identifier
    well : str
        Well identifier
    data_type : str
        Data type for plot naming
    output_dir : Path
        Output directory for saving plot

    Returns:
    -------
    Path or None
        Path to created QC plot, or None if creation failed
    """
    if len(cell_positions) > 0 and "tile" in cell_positions.columns:
        try:
            qc_path = output_dir / f"P-{plate}_W-{well}__{data_type}_tile_qc.png"
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            create_tile_arrangement_qc_plot(cell_positions, str(qc_path), data_type)
            print_success(f"QC plot saved: {qc_path}")
            return qc_path
        except Exception as e:
            print_warning(f"QC plot creation failed: {e}")
            return None
    else:
        print_warning("Skipping QC plot - insufficient data")
        return None


def main():
    """Main execution function for Snakemake script.

    This function orchestrates the complete stitching workflow, handling
    input validation, processing, and output generation according to the
    Snakemake rule configuration.
    """
    data_type = snakemake.params.data_type
    plate = snakemake.params.plate
    well = snakemake.params.well

    print_progress(f"STITCHING {data_type.upper()} - Plate {plate}, Well {well}")

    try:
        # Load and validate inputs
        print_progress("Loading metadata...")
        metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))

        # Apply SBS-specific filtering if needed
        if data_type == "sbs":
            sbs_filters = {"cycle": snakemake.config["merge"]["sbs_metadata_cycle"]}
            for filter_key, filter_value in sbs_filters.items():
                metadata = metadata[metadata[filter_key] == filter_value]
            print_progress(f"Applied SBS filter: {len(metadata)} entries remaining")

        # Load stitching configuration
        print_progress("Loading stitch configuration...")
        with open(snakemake.input[1], "r") as f:
            stitch_config = yaml.safe_load(f)

        # Process stitching workflow
        results = process_stitching(
            metadata, stitch_config, plate, well, data_type, snakemake.params
        )

        # Save outputs
        print_progress("Saving outputs...")

        # Save stitched image
        np.save(snakemake.output[0], results["stitched_image"])
        print_success(f"Stitched image saved: {snakemake.output[0]}")

        # Save stitched mask
        np.save(snakemake.output[1], results["stitched_mask"])
        print_success(f"Stitched mask saved: {snakemake.output[1]}")

        # Save cell positions
        results["cell_positions"].to_parquet(snakemake.output[2])
        print_success(
            f"Cell positions saved: {snakemake.output[2]} "
            f"({len(results['cell_positions'])} cells)"
        )

        # Create optional QC plot (if 4th output defined)
        if len(snakemake.output) > 3:
            qc_output_dir = Path(snakemake.output[3]).parent
            create_qc_plot(
                results["cell_positions"], plate, well, data_type, qc_output_dir
            )

        print_success(f"{data_type.upper()} stitching completed successfully!")

    except Exception as e:
        print_error(f"Stitching failed: {e}")
        raise


if __name__ == "__main__":
    main()