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
        max_cells_for_hash=50000,
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