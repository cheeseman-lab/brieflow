import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import stitched_well_alignment, merge_stitched_cells

# Load cell positions from stitched masks
phenotype_positions = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_positions))
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

# Load metadata files
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_metadata))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))

# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well

print(f"=== Starting Metadata-Aware Well-Level Merge ===")
print(f"Plate {plate}, Well {well}")
print(f"Phenotype cells: {len(phenotype_positions):,}")
print(f"SBS cells: {len(sbs_positions):,}")
print(f"Total cells: {len(phenotype_positions) + len(sbs_positions):,}")

# Extract pixel sizes from metadata
print(f"\n=== Extracting Pixel Sizes from Metadata ===")

# Filter metadata to this plate and well
phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata["plate"] == int(plate)) & 
    (phenotype_metadata["well"] == well)
]

sbs_well_metadata = sbs_metadata[
    (sbs_metadata["plate"] == int(plate)) & 
    (sbs_metadata["well"] == well)
]

# Extract pixel size for phenotype
phenotype_pixel_size = None
if (
    "pixel_size_x" in phenotype_well_metadata.columns
    and "pixel_size_y" in phenotype_well_metadata.columns
    and len(phenotype_well_metadata) > 0
):
    phenotype_pixel_size = phenotype_well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
    print(f"Phenotype pixel size from metadata: {phenotype_pixel_size:.6f} μm/pixel")
    
    # Verify pixel_size_y matches pixel_size_x
    pixel_size_y = phenotype_well_metadata["pixel_size_y"].iloc[0]
    if abs(phenotype_pixel_size - pixel_size_y) > 1e-6:
        print(f"⚠️  Warning: phenotype pixel_size_x ({phenotype_pixel_size:.6f}) != pixel_size_y ({pixel_size_y:.6f})")
else:
    print("⚠️  No pixel size found in phenotype metadata")

# Extract pixel size for SBS
sbs_pixel_size = None
if (
    "pixel_size_x" in sbs_well_metadata.columns
    and "pixel_size_y" in sbs_well_metadata.columns
    and len(sbs_well_metadata) > 0
):
    sbs_pixel_size = sbs_well_metadata["pixel_size_x"].iloc[0]  # μm per pixel
    print(f"SBS pixel size from metadata: {sbs_pixel_size:.6f} μm/pixel")
    
    # Verify pixel_size_y matches pixel_size_x
    pixel_size_y = sbs_well_metadata["pixel_size_y"].iloc[0]
    if abs(sbs_pixel_size - pixel_size_y) > 1e-6:
        print(f"⚠️  Warning: SBS pixel_size_x ({sbs_pixel_size:.6f}) != pixel_size_y ({pixel_size_y:.6f})")
else:
    print("⚠️  No pixel size found in SBS metadata")

# Fall back to typical values if metadata not available
if phenotype_pixel_size is None:
    phenotype_pixel_size = 0.325  # Typical for 40x
    print(f"Using typical 40x pixel size: {phenotype_pixel_size:.3f} μm/pixel")

if sbs_pixel_size is None:
    sbs_pixel_size = 1.30  # Typical for 10x
    print(f"Using typical 10x pixel size: {sbs_pixel_size:.3f} μm/pixel")

# Calculate expected scale factor
expected_scale_factor = sbs_pixel_size / phenotype_pixel_size
print(f"Expected scale factor (SBS/phenotype): {expected_scale_factor:.3f}")

# Analyze coordinate ranges to detect if stitching normalized scales
print(f"\n=== Analyzing Coordinate Ranges ===")
phenotype_i_range = phenotype_positions['i'].max() - phenotype_positions['i'].min()
phenotype_j_range = phenotype_positions['j'].max() - phenotype_positions['j'].min()
sbs_i_range = sbs_positions['i'].max() - sbs_positions['i'].min()
sbs_j_range = sbs_positions['j'].max() - sbs_positions['j'].min()

print(f"Coordinate ranges:")
print(f"  Phenotype: i={phenotype_i_range:.0f}, j={phenotype_j_range:.0f}")
print(f"  SBS: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")

# Calculate empirical scale factor from coordinates
if phenotype_i_range > 0 and sbs_i_range > 0:
    empirical_scale_i = sbs_i_range / phenotype_i_range
    empirical_scale_j = sbs_j_range / phenotype_j_range
    empirical_scale = (empirical_scale_i + empirical_scale_j) / 2
    
    print(f"Empirical scale factor from coordinates: {empirical_scale:.3f}")
    print(f"Expected scale factor from pixel sizes: {expected_scale_factor:.3f}")
    print(f"Scale difference: {abs(empirical_scale - expected_scale_factor):.3f}")
    
    # Check if stitching normalized coordinates
    if abs(empirical_scale - 1.0) < 0.2:
        print("→ Stitching appears to have normalized coordinates to same scale")
        use_normalized_scale = True
        effective_sbs_pixel_size = phenotype_pixel_size  # Same scale after stitching
    elif abs(empirical_scale - expected_scale_factor) < 0.5:
        print("→ Stitching preserved original pixel scales")
        use_normalized_scale = False
        effective_sbs_pixel_size = sbs_pixel_size
    else:
        print("→ Uncertain about stitching scale normalization, using empirical scale")
        use_normalized_scale = False
        effective_sbs_pixel_size = phenotype_pixel_size * empirical_scale
        
    print(f"Using effective pixel sizes: phenotype={phenotype_pixel_size:.6f}, SBS={effective_sbs_pixel_size:.6f}")
else:
    use_normalized_scale = False
    effective_sbs_pixel_size = sbs_pixel_size

# Check if we have sufficient data
if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
    print("Insufficient cells for triangulation and alignment")
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    print("Created empty merge results")
    exit(0)

# Filter to the specific well if not already filtered
if "well" in phenotype_positions.columns:
    phenotype_well = phenotype_positions[phenotype_positions["well"] == well]
else:
    phenotype_well = phenotype_positions

if "well" in sbs_positions.columns:
    sbs_well = sbs_positions[sbs_positions["well"] == well]
else:
    sbs_well = sbs_positions

print(f"After filtering: {len(phenotype_well):,} phenotype cells, {len(sbs_well):,} SBS cells")

if len(phenotype_well) == 0 or len(sbs_well) == 0:
    print("No cells found for this well after filtering")
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

# Perform metadata-aware alignment
print("\n=== Performing Metadata-Aware Triangle Hash Alignment ===")
try:
    alignment_df = stitched_well_alignment(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        det_range=snakemake.params.det_range,
        score_threshold=snakemake.params.score,
        max_cells_for_hash=75000,
        triangle_distance_threshold=0.3,
        min_matching_triangles=10,
        phenotype_pixel_size=phenotype_pixel_size,
        sbs_pixel_size=effective_sbs_pixel_size
    )

    if len(alignment_df) == 0:
        print("❌ Alignment failed - no valid alignment found")
        empty_merge = pd.DataFrame(columns=[
            "plate", "well", "cell_0", "i_0", "j_0", "area_0",
            "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
        ])
        empty_merge.to_parquet(snakemake.output[0])
        exit(0)

    alignment = alignment_df.iloc[0]
    print(f"✅ Alignment successful:")
    print(f"   Score: {alignment['score']:.3f}")
    print(f"   Determinant: {alignment['determinant']:.3f}")
    print(f"   Matched triangles: {alignment['n_triangles_matched']}")
    print(f"   Expected scale factor: {alignment.get('expected_scale_factor', 'N/A')}")
    print(f"   Scale normalized: {alignment.get('scale_normalized', 'N/A')}")
    print(f"   Cells used for alignment: {alignment['cells_used_phenotype']} phenotype, {alignment['cells_used_sbs']} SBS")

except Exception as e:
    print(f"❌ Alignment failed with error: {e}")
    import traceback
    traceback.print_exc()
    
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

# Check alignment quality with transformation-type-aware bounds
det_lower, det_upper = snakemake.params.det_range
score_min = snakemake.params.score

# Get transformation type from alignment result
transformation_type = alignment.get('transformation_type', 'unknown')

print(f"Alignment transformation type: {transformation_type}")

# Adjust determinant bounds based on transformation type
if transformation_type in ['translation_only', 'translation_fallback', 'translation_only_fallback']:
    # For translation-only transformations, determinant should be 1.0 (identity rotation)
    adjusted_det_lower = 0.9
    adjusted_det_upper = 1.1
    print(f"Using translation-only determinant bounds: [{adjusted_det_lower:.3f}, {adjusted_det_upper:.3f}]")
    
elif transformation_type in ['translation_rotation']:
    # For translation + rotation, determinant should be close to 1.0 but allow some variation
    adjusted_det_lower = 0.5
    adjusted_det_upper = 2.0
    print(f"Using translation+rotation determinant bounds: [{adjusted_det_lower:.3f}, {adjusted_det_upper:.3f}]")
    
elif 'expected_scale_factor' in alignment and alignment['expected_scale_factor'] is not None:
    # For full transformations with known scale factors
    scale_factor = alignment['expected_scale_factor']
    adjusted_det_lower = max(0.1, scale_factor * 0.3)
    adjusted_det_upper = min(20.0, scale_factor * 3.0)
    print(f"Using scale-aware determinant bounds for scale factor {scale_factor:.3f}: [{adjusted_det_lower:.3f}, {adjusted_det_upper:.3f}]")
    
elif use_normalized_scale:
    # If stitching normalized coordinates, expect determinant near 1
    adjusted_det_lower = 0.5
    adjusted_det_upper = 2.0
    print(f"Using normalized coordinate determinant bounds: [{adjusted_det_lower:.3f}, {adjusted_det_upper:.3f}]")
    
else:
    # Default bounds - but make them more permissive than original
    adjusted_det_lower = 0.1
    adjusted_det_upper = 10.0
    print(f"Using permissive default determinant bounds: [{adjusted_det_lower:.3f}, {adjusted_det_upper:.3f}]")

# Check alignment quality
alignment_passed = True
failure_reasons = []

if (alignment["determinant"] < adjusted_det_lower or alignment["determinant"] > adjusted_det_upper):
    alignment_passed = False
    failure_reasons.append(f"Determinant: {alignment['determinant']:.3f} (required: {adjusted_det_lower:.3f}-{adjusted_det_upper:.3f})")

if alignment["score"] < score_min:
    alignment_passed = False
    failure_reasons.append(f"Score: {alignment['score']:.3f} (required: >{score_min})")

if not alignment_passed:
    print(f"⚠️  Alignment quality insufficient:")
    for reason in failure_reasons:
        print(f"   {reason}")

    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

print(f"✅ Alignment quality check passed:")
print(f"   Determinant: {alignment['determinant']:.3f} (within bounds {adjusted_det_lower:.3f}-{adjusted_det_upper:.3f})")
print(f"   Score: {alignment['score']:.3f} (above threshold {score_min})")
print(f"   Transformation type: {transformation_type}")
# ADD THIS ENTIRE SECTION TO THE END OF YOUR SCRIPT:

print(f"\n=== DEBUG: About to start cell merging ===")
print(f"Phenotype well cells: {len(phenotype_well)}")
print(f"SBS well cells: {len(sbs_well)}")
print(f"Alignment object type: {type(alignment)}")
print(f"Output file path: {snakemake.output[0]}")

import os
output_dir = os.path.dirname(snakemake.output[0])
print(f"Output directory: {output_dir}")
print(f"Output directory exists: {os.path.exists(output_dir)}")
if not os.path.exists(output_dir):
    print("Creating output directory...")
    os.makedirs(output_dir, exist_ok=True)

# Try running with these improved parameters in your script:

# 1. First, let's add some diagnostics before merging:
print(f"\n=== Pre-Merge Diagnostics ===")
print(f"Alignment translation: {alignment['translation']}")
translation = alignment['translation']

# Check coordinate ranges after transformation
phenotype_coords = phenotype_well[['i', 'j']].values
sbs_coords = sbs_well[['i', 'j']].values

# Apply the transformation to see overlap
transformed_phenotype = phenotype_coords + translation

print(f"Original coordinate ranges:")
print(f"  Phenotype: i=[{phenotype_coords[:, 0].min():.0f}, {phenotype_coords[:, 0].max():.0f}], j=[{phenotype_coords[:, 1].min():.0f}, {phenotype_coords[:, 1].max():.0f}]")
print(f"  SBS: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")

print(f"After transformation:")
print(f"  Transformed phenotype: i=[{transformed_phenotype[:, 0].min():.0f}, {transformed_phenotype[:, 0].max():.0f}], j=[{transformed_phenotype[:, 1].min():.0f}, {transformed_phenotype[:, 1].max():.0f}]")

# Calculate overlap regions
overlap_i_min = max(transformed_phenotype[:, 0].min(), sbs_coords[:, 0].min())
overlap_i_max = min(transformed_phenotype[:, 0].max(), sbs_coords[:, 0].max())
overlap_j_min = max(transformed_phenotype[:, 1].min(), sbs_coords[:, 1].min())
overlap_j_max = min(transformed_phenotype[:, 1].max(), sbs_coords[:, 1].max())

overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min) if overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min else 0

print(f"Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
print(f"Overlap area: {overlap_area:.0f} square pixels")

if overlap_area <= 0:
    print("⚠️  WARNING: No coordinate overlap detected after transformation!")
    print("   This could explain the low merge rate.")

# Count cells in overlap region
phenotype_in_overlap = ((transformed_phenotype[:, 0] >= overlap_i_min) & (transformed_phenotype[:, 0] <= overlap_i_max) & 
                       (transformed_phenotype[:, 1] >= overlap_j_min) & (transformed_phenotype[:, 1] <= overlap_j_max)).sum()

sbs_in_overlap = ((sbs_coords[:, 0] >= overlap_i_min) & (sbs_coords[:, 0] <= overlap_i_max) & 
                  (sbs_coords[:, 1] >= overlap_j_min) & (sbs_coords[:, 1] <= overlap_j_max)).sum()

print(f"Cells in overlap region:")
print(f"  Phenotype: {phenotype_in_overlap:,} ({100*phenotype_in_overlap/len(phenotype_well):.1f}%)")
print(f"  SBS: {sbs_in_overlap:,} ({100*sbs_in_overlap/len(sbs_well):.1f}%)")

expected_max_matches = min(phenotype_in_overlap, sbs_in_overlap)
print(f"Expected max possible matches: {expected_max_matches:,}")

if expected_max_matches > 0:
    actual_merge_rate = 250 / expected_max_matches
    print(f"Actual merge rate within overlap: {actual_merge_rate:.3f} ({100*actual_merge_rate:.1f}%)")

# Merge cells using memory-efficient approach
print("\n=== Merging Cells ===")
try:
    merged_cells = merge_stitched_cells(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        alignment=alignment,
        threshold=snakemake.params.threshold,
        chunk_size=50000
    )
    print(f"✅ Cell merging completed successfully")
    print(f"Merged cells type: {type(merged_cells)}")
    print(f"Merged cells length: {len(merged_cells)}")

    if len(merged_cells) == 0:
        print("⚠️  No cells merged (no matches within distance threshold)")
    else:
        print(f"✅ Successfully merged {len(merged_cells):,} cells")
        print(f"   Distance threshold: {snakemake.params.threshold}")
        print(f"   Mean distance: {merged_cells['distance'].mean():.2f}")
        print(f"   Max distance: {merged_cells['distance'].max():.2f}")

    # Add metadata
    if "plate" not in merged_cells.columns or merged_cells["plate"].isna().all():
        merged_cells["plate"] = int(plate)
    merged_cells["well"] = well

    # Add tile and site columns for downstream compatibility
    print("\n=== Adding Tile/Site Compatibility ===")
    if "tile" not in merged_cells.columns:
        merged_cells["tile"] = 1
        print("✅ Using default tile=1 (stitched well-level data)")

    if "site" not in merged_cells.columns:
        merged_cells["site"] = 1
        print("✅ Using default site=1 (stitched well-level data)")

    # Verify critical columns
    required_columns = ["plate", "well", "cell_0", "cell_1", "tile", "site"]
    missing_required = [col for col in required_columns if col not in merged_cells.columns]

    if missing_required:
        print(f"❌ ERROR: Missing required columns: {missing_required}")
    else:
        print("✅ All required columns present for format_merge.py")

    # Show sample
    if len(merged_cells) > 0:
        sample_cols = ["plate", "well", "cell_0", "cell_1", "tile", "site", "distance"]
        print("Sample merged data:")
        print(merged_cells[sample_cols].head())

except Exception as e:
    print(f"❌ Cell merging failed with error: {e}")
    import traceback
    traceback.print_exc()
    
    merged_cells = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])

# Save results with debug output
print(f"\n=== DEBUG: About to save results ===")
print(f"Final merged_cells type: {type(merged_cells)}")
print(f"Final merged_cells length: {len(merged_cells)}")
print(f"Final merged_cells columns: {list(merged_cells.columns) if hasattr(merged_cells, 'columns') else 'No columns'}")
print(f"Output file path: {snakemake.output[0]}")

output_dir = os.path.dirname(snakemake.output[0])
print(f"Output directory exists: {os.path.exists(output_dir)}")
print(f"Output directory writable: {os.access(output_dir, os.W_OK) if os.path.exists(output_dir) else 'Directory does not exist'}")

# Save results with error handling
try:
    print("Attempting to save parquet file...")
    merged_cells.to_parquet(snakemake.output[0])
    print(f"✅ Successfully saved to: {snakemake.output[0]}")
    
    # Verify file was created
    if os.path.exists(snakemake.output[0]):
        file_size = os.path.getsize(snakemake.output[0])
        print(f"✅ Output file confirmed: {file_size} bytes")
    else:
        print(f"❌ Output file was not created!")
        
except Exception as e:
    print(f"❌ Failed to save parquet file: {e}")
    import traceback
    traceback.print_exc()
    
    # Try alternative save methods
    try:
        print("Trying to save as CSV instead...")
        csv_path = snakemake.output[0].replace('.parquet', '.csv')
        merged_cells.to_csv(csv_path)
        print(f"✅ Saved as CSV: {csv_path}")
    except Exception as e2:
        print(f"❌ CSV save also failed: {e2}")

print(f"\n=== Results Saved ===")
print(f"Output file: {snakemake.output[0]}")
print(f"Final merged cells: {len(merged_cells):,}")

if len(merged_cells) > 0:
    print(f"Distance statistics:")
    print(f"   Mean: {merged_cells['distance'].mean():.3f}")
    print(f"   Std:  {merged_cells['distance'].std():.3f}")
    print(f"   Min:  {merged_cells['distance'].min():.3f}")
    print(f"   Max:  {merged_cells['distance'].max():.3f}")

print("Metadata-aware well-level merge completed successfully!")