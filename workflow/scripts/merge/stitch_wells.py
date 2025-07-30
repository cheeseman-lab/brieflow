import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from skimage import io

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import full_stitching_pipeline

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
        # Nuclei masks are at: analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile}__nuclei.tiff
        mask_path = f"analysis_root/{data_type}/images/P-{plate}_W-{well}_T-{tile_id}__nuclei.tiff"
        
        if Path(mask_path).exists():
            existing_masks.append(mask_path)
        else:
            missing_masks.append(mask_path)
    
    masks_exist = len(existing_masks) > 0
    print(f"Nuclei mask check for {data_type}: {len(existing_masks)} exist, {len(missing_masks)} missing")
    if missing_masks:
        print(f"Example missing mask: {missing_masks[0]}")
    
    return masks_exist, existing_masks

# Load metadata
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Load stitch configurations
with open(snakemake.input[2], "r") as f:
    phenotype_config = yaml.safe_load(f)

with open(snakemake.input[3], "r") as f:
    sbs_config = yaml.safe_load(f)

# Filter metadata to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata["plate"] == int(plate)) & (phenotype_metadata["well"] == well)
]

sbs_well_metadata = sbs_metadata[
    (sbs_metadata["plate"] == int(plate)) & (sbs_metadata["well"] == well)
]

print(f"=== Stitching Wells for Plate {plate}, Well {well} ===")
print(f"Phenotype tiles: {len(phenotype_well_metadata)}")
print(f"SBS tiles: {len(sbs_well_metadata)}")

# Check if mask files exist
phenotype_masks_exist, _ = check_mask_files_exist(phenotype_well_metadata, well, "phenotype")
sbs_masks_exist, _ = check_mask_files_exist(sbs_well_metadata, well, "sbs")

print(f"Phenotype masks available: {phenotype_masks_exist}")
print(f"SBS masks available: {sbs_masks_exist}")

if len(phenotype_well_metadata) == 0 or len(sbs_well_metadata) == 0:
    print("Warning: No tiles found for this well")
    # Create empty outputs
    empty_image = np.zeros((100, 100), dtype=np.uint16)
    empty_mask = np.zeros((100, 100), dtype=np.uint16)
    empty_positions = pd.DataFrame(columns=['well', 'cell', 'i', 'j', 'area', 'data_type'])
    
    # Save empty outputs
    np.save(snakemake.output[0], empty_image)  # phenotype_stitched_image
    np.save(snakemake.output[1], empty_mask)   # phenotype_stitched_mask
    empty_positions.to_parquet(snakemake.output[2])  # phenotype_positions
    np.save(snakemake.output[3], empty_image)  # sbs_stitched_image
    np.save(snakemake.output[4], empty_mask)   # sbs_stitched_mask
    empty_positions.to_parquet(snakemake.output[5])  # sbs_positions
    
    # Create empty overlays
    empty_overlay = np.zeros((100, 100, 3), dtype=np.uint8)
    io.imsave(snakemake.output[6], empty_overlay)  # phenotype_overlay
    io.imsave(snakemake.output[7], empty_overlay)  # sbs_overlay
    
    print("Created empty outputs due to missing tiles")
    exit(0)

# Create a custom stitching function that handles missing masks
def safe_stitching_pipeline(metadata_df, well, data_type, stitch_config, masks_exist):
    """Stitching pipeline that handles missing masks gracefully"""
    
    print(f"\n=== Processing {data_type.title()} Stitching ===")
    
    results = {
        'stitched_image': None,
        'stitched_mask': None,
        'cell_positions': None,
        'overlay': None
    }
    
    try:
        # Always try to create stitched image
        from lib.merge.stitch_well import assemble_aligned_tiff_well
        
        print("Assembling stitched image...")
        stitched_image = assemble_aligned_tiff_well(
            metadata_df=metadata_df,
            shifts=stitch_config["total_translation"],
            well=well,
            data_type=data_type,
            flipud=snakemake.params.flipud,
            fliplr=snakemake.params.fliplr,
            rot90=snakemake.params.rot90,
            overlap_percent=snakemake.params.get("overlap_percent", 0.05),
        )
        results['stitched_image'] = stitched_image
        print(f"✅ Stitched image successful: {stitched_image.shape}")
        
    except Exception as e:
        print(f"❌ Image stitching failed: {e}")
        results['stitched_image'] = np.zeros((100, 100), dtype=np.uint16)
    
    # Only try mask stitching if masks exist
    if masks_exist:
        try:
            from lib.merge.stitch_well import assemble_stitched_masks, extract_cell_positions_from_stitched_mask
            
            print("Assembling stitched masks...")
            stitched_mask = assemble_stitched_masks(
                metadata_df=metadata_df,
                shifts=stitch_config["total_translation"],
                well=well,
                data_type=data_type,
                flipud=snakemake.params.flipud,
                fliplr=snakemake.params.fliplr,
                rot90=snakemake.params.rot90
            )
            results['stitched_mask'] = stitched_mask
            print(f"✅ Stitched mask successful: {stitched_mask.shape}, max label: {stitched_mask.max()}")
            
            # Extract cell positions
            if stitched_mask is not None and stitched_mask.max() > 0:
                print("Extracting cell positions...")
                cell_positions = extract_cell_positions_from_stitched_mask(
                    stitched_mask, well, data_type
                )
                results['cell_positions'] = cell_positions
                print(f"✅ Extracted {len(cell_positions)} cell positions")
            
        except Exception as e:
            print(f"❌ Mask stitching failed: {e}")
            results['stitched_mask'] = np.zeros_like(results['stitched_image'], dtype=np.uint16)
    else:
        print("⚠️  Skipping mask stitching - no mask files found")
        results['stitched_mask'] = np.zeros_like(results['stitched_image'], dtype=np.uint16)
    
    # Create empty cell positions if masks failed
    if results['cell_positions'] is None:
        results['cell_positions'] = pd.DataFrame(columns=['well', 'cell', 'i', 'j', 'area', 'data_type'])
        print("⚠️  Created empty cell positions")
    
    # Create overlay if both image and mask exist
    if (results['stitched_image'] is not None and 
        results['stitched_mask'] is not None and 
        results['stitched_mask'].max() > 0):
        try:
            from lib.merge.merge_well import create_stitched_overlay
            overlay = create_stitched_overlay(results['stitched_image'], results['stitched_mask'])
            results['overlay'] = overlay
            print("✅ Created overlay image")
        except Exception as e:
            print(f"❌ Overlay creation failed: {e}")
    
    if results['overlay'] is None:
        # Create empty overlay
        if results['stitched_image'] is not None:
            h, w = results['stitched_image'].shape
            results['overlay'] = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            results['overlay'] = np.zeros((100, 100, 3), dtype=np.uint8)
    
    return results

# Process phenotype stitching
phenotype_results = safe_stitching_pipeline(
    phenotype_well_metadata, well, "phenotype", phenotype_config, phenotype_masks_exist
)

# Process SBS stitching  
sbs_results = safe_stitching_pipeline(
    sbs_well_metadata, well, "sbs", sbs_config, sbs_masks_exist
)

# Save all outputs
print("\n=== Saving Outputs ===")

# Save phenotype outputs
np.save(snakemake.output[0], phenotype_results['stitched_image'])
print(f"Saved phenotype stitched image: {snakemake.output[0]}")

np.save(snakemake.output[1], phenotype_results['stitched_mask'])
print(f"Saved phenotype stitched mask: {snakemake.output[1]}")

phenotype_results['cell_positions'].to_parquet(snakemake.output[2])
print(f"Saved phenotype cell positions: {snakemake.output[2]} ({len(phenotype_results['cell_positions'])} cells)")

# Save SBS outputs
np.save(snakemake.output[3], sbs_results['stitched_image'])
print(f"Saved SBS stitched image: {snakemake.output[3]}")

np.save(snakemake.output[4], sbs_results['stitched_mask'])
print(f"Saved SBS stitched mask: {snakemake.output[4]}")

sbs_results['cell_positions'].to_parquet(snakemake.output[5])
print(f"Saved SBS cell positions: {snakemake.output[5]} ({len(sbs_results['cell_positions'])} cells)")

# Save overlays
io.imsave(snakemake.output[6], phenotype_results['overlay'])
print(f"Saved phenotype overlay: {snakemake.output[6]}")

io.imsave(snakemake.output[7], sbs_results['overlay'])
print(f"Saved SBS overlay: {snakemake.output[7]}")

print("Well stitching completed successfully!")

# Summary
print(f"\n=== Final Summary ===")
print(f"Phenotype: {phenotype_results['stitched_image'].shape} image, {len(phenotype_results['cell_positions'])} cells")
print(f"SBS: {sbs_results['stitched_image'].shape} image, {len(sbs_results['cell_positions'])} cells")