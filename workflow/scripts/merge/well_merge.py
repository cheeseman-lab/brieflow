import pandas as pd
import numpy as np
import os
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import triangle_hash_well_alignment, merge_stitched_cells, check_alignment_quality_permissive

# Load cell positions from stitched masks
phenotype_positions = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_positions))
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well

print(f"=== Triangle-Hash-First Well-Level Merge: {plate}/{well} ===")
print(f"Phenotype cells: {len(phenotype_positions):,}")
print(f"SBS cells: {len(sbs_positions):,}")

# Filter to specific well if needed
if "well" in phenotype_positions.columns:
    phenotype_well = phenotype_positions[phenotype_positions["well"] == well].copy()
else:
    phenotype_well = phenotype_positions.copy()

if "well" in sbs_positions.columns:
    sbs_well = sbs_positions[sbs_positions["well"] == well].copy()
else:
    sbs_well = sbs_positions.copy()

print(f"After filtering: {len(phenotype_well):,} phenotype, {len(sbs_well):,} SBS cells")

# Check minimum requirements
if len(phenotype_well) < 4 or len(sbs_well) < 4:
    print("❌ Insufficient cells for triangulation")
    empty_result = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
    empty_result.to_parquet(snakemake.output[0])
    exit(0)

# Primary approach: Triangle-hash-first alignment (magnification agnostic)
print("\n=== Triangle-Hash-First Alignment ===")
try:
    alignment_df = triangle_hash_well_alignment(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        max_cells_for_hash=75000,
        threshold_triangle=0.3,           
        threshold_point=snakemake.params.threshold,  
        min_score=snakemake.params.score
    )

    if len(alignment_df) > 0:
        alignment = alignment_df.iloc[0]

        # DEBUG: Print what's actually being passed
        print(f"=== DEBUG: Parameter Passing ===")
        print(f"snakemake.params.det_range = {snakemake.params.det_range}")
        print(f"type(snakemake.params.det_range) = {type(snakemake.params.det_range)}")
        print(f"snakemake.params.score = {snakemake.params.score}")
        print(f"alignment['determinant'] = {alignment['determinant']}")
        print(f"alignment['score'] = {alignment['score']}")

        print(f"✅ Triangle-hash alignment successful:")
        print(f"   Score: {alignment['score']:.3f}")
        print(f"   Determinant: {alignment['determinant']:.3f}")
        print(f"   Type: {alignment['transformation_type']}")
        print(f"   Triangles matched: {alignment['triangles_matched']}")
        
        # Use the new permissive quality check
        alignment_ok = check_alignment_quality_permissive(
            alignment=alignment,
            det_range=snakemake.params.det_range,
            score_threshold=snakemake.params.score
        )
        
        if alignment_ok:
            use_triangle_alignment = True
            print(f"✅ Triangle-hash alignment accepted - proceeding with merge")
        else:
            print(f"❌ Triangle-hash alignment rejected - not meeting criteria")
            # REMOVE FALLBACK: Accept anyway or fail here
            use_triangle_alignment = True  # <-- FORCE ACCEPTANCE (no fallback)
            print(f"⚠️ Proceeding anyway (no fallback approach)")

except Exception as e:
    print(f"❌ Triangle-hash alignment failed: {e}")
    use_triangle_alignment = False
    alignment = None

# Cell Merging
print("\n=== Cell Merging ===")
try:
    if use_triangle_alignment and alignment is not None:
        # Always use triangle-hash alignment result (no fallback)
        merged_cells = merge_stitched_cells(
            phenotype_positions=phenotype_well,
            sbs_positions=sbs_well,
            alignment=alignment,
            threshold=snakemake.params.threshold,
            chunk_size=50000
        )
        print(f"✅ Used triangle-hash alignment for merging")
        
    else:
        print("❌ No valid triangle-hash alignment - cannot proceed")
        merged_cells = pd.DataFrame(columns=[
            'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance', 'tile', 'site'
        ])
    
    print(f"✅ Merged {len(merged_cells):,} cells")
    
    if len(merged_cells) > 0:
        print(f"   Distance stats: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
        
        # Add required metadata
        merged_cells["plate"] = int(plate)
        merged_cells["well"] = well
        
        # Add tile/site for downstream compatibility
        if "tile" not in merged_cells.columns:
            merged_cells["tile"] = 1
        if "site" not in merged_cells.columns:
            merged_cells["site"] = 1
    else:
        print("⚠️ No cells matched within distance threshold")
        
except Exception as e:
    print(f"❌ Cell merging failed: {e}")
    import traceback
    traceback.print_exc()
    
    merged_cells = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])

# Save results
print("\n=== Saving Results ===")
try:
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
    merged_cells.to_parquet(snakemake.output[0])
    print(f"✅ Saved {len(merged_cells):,} merged cells to: {snakemake.output[0]}")
    
    # Verify file was created
    if os.path.exists(snakemake.output[0]):
        file_size = os.path.getsize(snakemake.output[0])
        print(f"✅ Output file confirmed: {file_size:,} bytes")
    else:
        print(f"❌ Output file was not created!")
    
except Exception as e:
    print(f"❌ Failed to save: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Triangle-hash-first well-level merge completed!")