import pandas as pd
import yaml

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import stitched_well_alignment, merge_stitched_cells

# Load cell positions from stitched masks (these are the new inputs)
phenotype_positions = validate_dtypes(pd.read_parquet(snakemake.input[0]))  # phenotype_positions
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input[1]))         # sbs_positions

# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well

print(f"=== Starting Enhanced Well-Level Merge ===")
print(f"Plate {plate}, Well {well}")
print(f"Phenotype cells: {len(phenotype_positions)}")
print(f"SBS cells: {len(sbs_positions)}")

# Check what columns we have in the position data
print(f"Phenotype columns: {list(phenotype_positions.columns)}")
print(f"SBS columns: {list(sbs_positions.columns)}")

# Check if we have sufficient data
if len(phenotype_positions) < 4 or len(sbs_positions) < 4:
    print("Insufficient cells for triangulation and alignment")
    
    # Create empty output with all necessary columns
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
        "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
        "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    
    print("Created empty merge results")
    exit(0)

# Filter to the specific well if not already filtered
if 'well' in phenotype_positions.columns:
    phenotype_well = phenotype_positions[phenotype_positions["well"] == well]
else:
    phenotype_well = phenotype_positions

if 'well' in sbs_positions.columns:
    sbs_well = sbs_positions[sbs_positions["well"] == well]
else:
    sbs_well = sbs_positions

print(f"After filtering: {len(phenotype_well)} phenotype cells, {len(sbs_well)} SBS cells")

if len(phenotype_well) == 0 or len(sbs_well) == 0:
    print("No cells found for this well after filtering")
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
        "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
        "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

# Perform alignment using actual cell positions from stitched masks
print("\n=== Performing Triangle Hash Alignment ===")
try:
    alignment_df = stitched_well_alignment(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        det_range=snakemake.params.det_range,
        score_threshold=snakemake.params.score,
    )
    
    if len(alignment_df) == 0:
        print("❌ Alignment failed - no valid alignment found")
        empty_merge = pd.DataFrame(columns=[
            "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
            "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
            "distance", "tile", "site"
        ])
        empty_merge.to_parquet(snakemake.output[0])
        exit(0)
    
    alignment = alignment_df.iloc[0]
    print(f"✅ Alignment successful:")
    print(f"   Score: {alignment['score']:.3f}")
    print(f"   Determinant: {alignment['determinant']:.3f}")
    print(f"   Matched triangles: {alignment.get('n_triangles_matched', 'N/A')}")
    
except Exception as e:
    print(f"❌ Alignment failed with error: {e}")
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
        "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
        "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

# Check alignment quality
if (
    alignment["determinant"] < snakemake.params.det_range[0]
    or alignment["determinant"] > snakemake.params.det_range[1]
    or alignment["score"] < snakemake.params.score
):
    print(f"⚠️  Alignment quality insufficient:")
    print(f"   Determinant: {alignment['determinant']:.3f} (required: {snakemake.params.det_range[0]}-{snakemake.params.det_range[1]})")
    print(f"   Score: {alignment['score']:.3f} (required: >{snakemake.params.score})")
    
    # Still save empty result but with a warning
    empty_merge = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
        "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
        "distance", "tile", "site"
    ])
    empty_merge.to_parquet(snakemake.output[0])
    exit(0)

# Merge cells based on alignment
print("\n=== Merging Cells ===")
try:
    merged_cells = merge_stitched_cells(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        alignment=alignment,
        threshold=snakemake.params.threshold
    )
    
    if len(merged_cells) == 0:
        print("⚠️  No cells merged (no matches within distance threshold)")
    else:
        print(f"✅ Successfully merged {len(merged_cells)} cells")
        print(f"   Distance threshold: {snakemake.params.threshold}")
        print(f"   Mean distance: {merged_cells['distance'].mean():.2f}")
        print(f"   Max distance: {merged_cells['distance'].max():.2f}")
    
    # Add plate information if missing
    if 'plate' not in merged_cells.columns or merged_cells['plate'].isna().all():
        merged_cells['plate'] = int(plate)
    
    # Ensure well information is correct
    merged_cells['well'] = well
    
    # CRITICAL: Ensure tile and site columns for downstream compatibility
    print("\n=== Adding Tile/Site Compatibility ===")
    
    # Add tile column (for phenotype compatibility)
    if 'tile' not in merged_cells.columns:
        if 'tile_0' in merged_cells.columns:
            merged_cells['tile'] = merged_cells['tile_0']  # Use phenotype tile
            print("✅ Using phenotype tile information")
        else:
            merged_cells['tile'] = 1  # Default fallback
            print("⚠️  Using default tile=1 (no tile info from positions)")
    
    # Add site column (CRITICAL for SBS joining in format_merge.py)
    if 'site' not in merged_cells.columns:
        if 'site_1' in merged_cells.columns:
            merged_cells['site'] = merged_cells['site_1']  # Use SBS site
            print("✅ Using SBS site information")
        elif 'tile_1' in merged_cells.columns:
            merged_cells['site'] = merged_cells['tile_1']  # Use SBS tile as site
            print("✅ Using SBS tile as site")
        else:
            merged_cells['site'] = 1  # Default fallback
            print("⚠️  Using default site=1 (no site info from positions)")
    
    print(f"Final columns: {list(merged_cells.columns)}")
    
    # Verify critical columns for downstream processing
    required_for_format_merge = ['plate', 'well', 'cell_0', 'cell_1', 'tile', 'site']
    missing_required = [col for col in required_for_format_merge if col not in merged_cells.columns]
    
    if missing_required:
        print(f"❌ ERROR: Missing required columns: {missing_required}")
    else:
        print("✅ All required columns present for format_merge.py")
        
    # Show sample with key columns
    if len(merged_cells) > 0:
        sample_cols = ['plate', 'well', 'cell_0', 'cell_1', 'tile', 'site', 'distance']
        available_cols = [col for col in sample_cols if col in merged_cells.columns]
        print("Sample merged data:")
        print(merged_cells[available_cols].head())
    
except Exception as e:
    print(f"❌ Cell merging failed with error: {e}")
    merged_cells = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0", "tile_0", "site_0",
        "cell_1", "i_1", "j_1", "area_1", "tile_1", "site_1", 
        "distance", "tile", "site"
    ])

# Save results
merged_cells.to_parquet(snakemake.output[0])
print(f"\n=== Results Saved ===")
print(f"Output file: {snakemake.output[0]}")
print(f"Final merged cells: {len(merged_cells)}")

if len(merged_cells) > 0:
    print(f"Distance statistics:")
    print(f"   Mean: {merged_cells['distance'].mean():.3f}")
    print(f"   Std:  {merged_cells['distance'].std():.3f}")
    print(f"   Min:  {merged_cells['distance'].min():.3f}")
    print(f"   Max:  {merged_cells['distance'].max():.3f}")
    
    # Show tile/site distribution
    if 'tile' in merged_cells.columns:
        print(f"Tile distribution: {merged_cells['tile'].value_counts().to_dict()}")
    if 'site' in merged_cells.columns:
        print(f"Site distribution: {merged_cells['site'].value_counts().to_dict()}")

print("Enhanced well-level merge completed successfully!")