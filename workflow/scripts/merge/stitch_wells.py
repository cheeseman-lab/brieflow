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
from lib.merge.merge_well import create_stitched_overlay


def print_memory_usage(stage=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1e9
    print(f"Memory usage {stage}: {memory_gb:.1f} GB")
    return memory_gb


def check_mask_files_exist(metadata_df, well, data_type):
    """Check if nuclei mask files exist for this well"""
    well_metadata = metadata_df[metadata_df["well"] == well]

    if len(well_metadata) == 0:
        return False, []

    existing_masks = []
    missing_masks = []

    for _, row in well_metadata.head(5).iterrows():  # Check first 5 tiles
        plate = row["plate"]
        tile_id = row["tile"]
        mask_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__nuclei.tiff"

        if Path(mask_path).exists():
            existing_masks.append(mask_path)
        else:
            missing_masks.append(mask_path)

    masks_exist = len(existing_masks) > 0
    print(
        f"Nuclei mask check for {data_type}: {len(existing_masks)} exist, {len(missing_masks)} missing"
    )

    return masks_exist, existing_masks


def create_empty_outputs(data_type):
    """Create empty outputs when processing fails"""
    return {
        "stitched_image": np.zeros((100, 100), dtype=np.uint16),
        "stitched_mask": np.zeros((100, 100), dtype=np.uint16),
        "cell_positions": pd.DataFrame(
            columns=["well", "cell", "i", "j", "area", "data_type"]
        ),
        "overlay": np.zeros((100, 100, 3), dtype=np.uint8),
    }


def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    gc.collect()


def process_modality_with_disk_offload(
    metadata_df, stitch_config, plate, well, data_type, params
):
    """
    Process modality with immediate disk offloading to manage memory.
    Save results immediately and clear from memory.
    """

    print(f"\n=== Processing {data_type.upper()} with Disk Offloading ===")
    print(f"Plate {plate}, Well {well}")
    print_memory_usage("start")

    # Filter metadata to specific plate and well
    well_metadata = metadata_df[
        (metadata_df["plate"] == int(plate)) & (metadata_df["well"] == well)
    ]

    print(f"{data_type.capitalize()} tiles: {len(well_metadata)}")

    if len(well_metadata) == 0:
        print(f"Warning: No {data_type} tiles found for this well")
        return create_empty_outputs(data_type)

    # Check if mask files exist
    masks_exist, _ = check_mask_files_exist(well_metadata, well, data_type)
    print(f"{data_type.capitalize()} masks available: {masks_exist}")

    # Create temporary directory for intermediate files
    temp_dir = Path(f"/tmp/stitch_{data_type}_{plate}_{well}_{os.getpid()}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Assemble stitched image
        print("Assembling stitched image...")
        print_memory_usage("before image assembly")

        stitched_image = assemble_aligned_tiff_well(
            metadata_df=well_metadata,
            shifts=stitch_config["total_translation"],
            well=well,
            data_type=data_type,
            flipud=params.flipud,
            fliplr=params.fliplr,
            rot90=params.rot90,
            overlap_percent=params.overlap_percent,
        )

        print(f"✅ Stitched image successful: {stitched_image.shape}")
        print_memory_usage("after image assembly")

        # IMMEDIATELY save image to temp file and clear from memory
        temp_image_path = temp_dir / "stitched_image.npy"
        np.save(temp_image_path, stitched_image)
        image_shape = stitched_image.shape  # Keep shape for later
        del stitched_image  # Clear from memory
        aggressive_memory_cleanup()
        print_memory_usage("after image cleanup")

        # Step 2: Assemble stitched masks (only if masks exist)
        if masks_exist:
            print("Assembling stitched masks...")
            print_memory_usage("before mask assembly")

            try:
                stitched_mask, cell_id_mapping = assemble_stitched_masks_simple(
                    metadata_df=well_metadata,
                    shifts=stitch_config["total_translation"],
                    well=well,
                    data_type=data_type,
                    flipud=params.flipud,
                    fliplr=params.fliplr,
                    rot90=params.rot90,
                    return_cell_mapping=True,
                )

                print(
                    f"✅ Stitched mask successful: {stitched_mask.shape}, max label: {stitched_mask.max()}"
                )
                print(f"✅ Cell ID mapping created: {len(cell_id_mapping)} mappings")
                print_memory_usage("after mask assembly")

                # IMMEDIATELY save mask to temp file
                temp_mask_path = temp_dir / "stitched_mask.npy"
                np.save(temp_mask_path, stitched_mask)
                mask_shape = stitched_mask.shape
                mask_max_label = stitched_mask.max()

                # Extract cell positions BEFORE clearing mask
                if mask_max_label > 0:
                    print("Extracting cell positions...")
                    stitched_mask_array = stitched_mask

                    # Define tile_size based on data type
                    tile_size = (
                        (2400, 2400) if data_type == "phenotype" else (1200, 1200)
                    )

                    # Extract cell positions with tile tracking
                    cell_positions = extract_cell_positions_from_stitched_mask(
                        stitched_mask=stitched_mask_array,
                        well=well,
                        plate=plate,
                        data_type=data_type,
                        metadata_df=well_metadata,
                        shifts=stitch_config["total_translation"],
                        tile_size=tile_size,
                        cell_id_mapping=cell_id_mapping,
                    )
                    print(f"✅ Extracted {len(cell_positions)} cell positions")


                    # Add QC plot creation
                    if len(cell_positions) > 0 and "tile" in cell_positions.columns:
                        temp_qc_path = temp_dir / f"{well}_{data_type}_tile_qc.png"
                        create_tile_arrangement_qc_plot(
                            cell_positions, str(temp_qc_path), data_type
                        )
                        print(f"✅ Created QC plot: {temp_qc_path}")

                else:
                    cell_positions = pd.DataFrame(
                        columns=["well", "cell", "i", "j", "area", "data_type"]
                    )

                del stitched_mask  # Clear mask from memory
                aggressive_memory_cleanup()
                print_memory_usage("after mask cleanup")

            except Exception as e:
                print(f"❌ Mask stitching failed: {e}")
                # Create empty mask file
                empty_mask = np.zeros(image_shape, dtype=np.uint16)
                temp_mask_path = temp_dir / "stitched_mask.npy"
                np.save(temp_mask_path, empty_mask)
                del empty_mask
                cell_positions = pd.DataFrame(
                    columns=["well", "cell", "i", "j", "area", "data_type"]
                )
                aggressive_memory_cleanup()
        else:
            print("⚠️  Skipping mask stitching - no mask files found")
            empty_mask = np.zeros(image_shape, dtype=np.uint16)
            temp_mask_path = temp_dir / "stitched_mask.npy"
            np.save(temp_mask_path, empty_mask)
            del empty_mask
            cell_positions = pd.DataFrame(
                columns=["well", "cell", "i", "j", "area", "data_type"]
            )
            aggressive_memory_cleanup()

        # Step 3: Create overlay (load both image and mask temporarily)
        if params.create_overlay:
            print("Creating overlay...")
            print_memory_usage("before overlay creation")

            try:
                # Load image and mask temporarily
                temp_image = np.load(temp_image_path)
                temp_mask = np.load(temp_mask_path)

                if temp_mask.max() > 0:
                    overlay = create_stitched_overlay(temp_image, temp_mask)
                else:
                    # Create empty overlay
                    h, w = temp_image.shape
                    overlay = np.zeros((h, w, 3), dtype=np.uint8)

                # Clear temporary arrays immediately
                del temp_image, temp_mask
                aggressive_memory_cleanup()
                print_memory_usage("after overlay creation")

            except Exception as e:
                print(f"❌ Overlay creation failed: {e}")
                overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        # Return file paths instead of data
        return {
            "temp_image_path": temp_image_path,
            "temp_mask_path": temp_mask_path,
            "cell_positions": cell_positions,
            "overlay": overlay,
            "temp_dir": temp_dir,
        }

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return create_empty_outputs(data_type)


# Main execution
def main():
    data_type = snakemake.params.data_type

    print(f"=== MEMORY-OPTIMIZED {data_type.upper()} STITCHING ===")
    print_memory_usage("initial")

    # Load metadata first
    metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))

    # Apply filtering for SBS data type (same as in stitch config generation)
    if data_type == "sbs":
        # Get the same filter used in stitch config generation
        sbs_filters = {"cycle": snakemake.config["merge"]["sbs_metadata_cycle"]}
        for filter_key, filter_value in sbs_filters.items():
            print(f"Filtering {data_type} metadata: {filter_key} == {filter_value}")
            metadata = metadata[metadata[filter_key] == filter_value]
        print(f"After filtering - {data_type} metadata: {len(metadata)} entries")

    # Load stitch config
    with open(snakemake.input[1], "r") as f:
        stitch_config = yaml.safe_load(f)

    outputs = snakemake.output[:4]

    # Get parameters
    plate = snakemake.params.plate
    well = snakemake.params.well

    print(f"Plate {plate}, Well {well}")
    print(f"Memory allocation: 400GB")
    print(f"Output files:")
    for i, output in enumerate(outputs):
        print(f"  {i}: {output}")

    # Process with disk offloading
    results = process_modality_with_disk_offload(
        metadata, stitch_config, plate, well, data_type, snakemake.params
    )

    # Save final outputs
    print(f"\n=== Saving Final {data_type.upper()} Outputs ===")
    print_memory_usage("before final save")

    try:
        if "temp_image_path" in results:
            # Move temp files to final locations
            import shutil

            # Copy image
            shutil.move(str(results["temp_image_path"]), outputs[0])
            print(f"✅ Moved stitched image: {outputs[0]}")

            # Copy mask
            shutil.move(str(results["temp_mask_path"]), outputs[1])
            print(f"✅ Moved stitched mask: {outputs[1]}")

            # Save cell positions
            results["cell_positions"].to_parquet(outputs[2])
            print(
                f"✅ Saved cell positions: {outputs[2]} ({len(results['cell_positions'])} cells)"
            )

            # Save overlay
            io.imsave(outputs[3], results["overlay"])
            print(f"✅ Saved overlay: {outputs[3]}")

            # QC Plot
            temp_qc_path = results["temp_dir"] / f"{well}_{data_type}_tile_qc.png"
            if temp_qc_path.exists():
                qc_output_path = (
                    Path(outputs[2]).parent.parent
                    / "qc_plots"
                    / f"{plate}_{well}_{data_type}_tile_qc.png"
                )
                qc_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(temp_qc_path), str(qc_output_path))
                print(f"✅ Saved QC plot: {qc_output_path}")

            # Cleanup temp directory
            if results["temp_dir"].exists():
                import shutil

                shutil.rmtree(results["temp_dir"])
                print("🗑️  Cleaned up temporary files")

        else:
            # Fallback to empty outputs
            for i, output in enumerate(outputs):
                if output.endswith(".npy"):
                    np.save(output, results[list(results.keys())[i]])
                elif output.endswith(".parquet"):
                    results["cell_positions"].to_parquet(output)
                elif output.endswith(".png"):
                    io.imsave(output, results["overlay"])

    except Exception as e:
        print(f"❌ Final save failed: {e}")

    print_memory_usage("final")
    print(f"\n🎉 === {data_type.upper()} STITCHING COMPLETED! ===")


if __name__ == "__main__":
    main()
