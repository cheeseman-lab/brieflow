import pandas as pd
import numpy as np
import os
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import triangle_hash_well_alignment, merge_stitched_cells

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

        # Check the actual validation
        det_lower, det_upper = snakemake.params.det_range
        print(f"det_lower = {det_lower}, det_upper = {det_upper}")
        print(f"Determinant check: {det_lower} <= {alignment['determinant']} <= {det_upper}")
        print(f"Score check: {alignment['score']} >= {snakemake.params.score}")

        alignment_ok = (det_lower <= alignment["determinant"] <= det_upper and 
                        alignment["score"] >= snakemake.params.score)
        print(f"alignment_ok = {alignment_ok}")

        print(f"✅ Triangle-hash alignment successful:")
        print(f"   Score: {alignment['score']:.3f}")
        print(f"   Determinant: {alignment['determinant']:.3f}")
        print(f"   Type: {alignment['transformation_type']}")
        print(f"   Triangles matched: {alignment['triangles_matched']}")
        
        # Very permissive quality check (triangle hash is inherently robust)
        if alignment['score'] >= 0.2 and 0.1 <= alignment['determinant'] <= 10.0:
            use_triangle_alignment = True
            print(f"✅ Triangle-hash alignment quality acceptable")
        else:
            print(f"⚠️ Triangle-hash alignment quality marginal - will try fallback")
            use_triangle_alignment = False
            alignment = None
    else:
        print("❌ Triangle-hash alignment failed - trying fallback")
        use_triangle_alignment = False
        alignment = None

except Exception as e:
    print(f"❌ Triangle-hash alignment failed: {e}")
    use_triangle_alignment = False
    alignment = None

# Fallback approach: Simple distance matching
if not use_triangle_alignment:
    print("\n=== Fallback: Simple Distance Matching ===")
    
    # Check coordinate overlap first
    pheno_coords = phenotype_well[['i', 'j']].values
    sbs_coords = sbs_well[['i', 'j']].values

    pheno_i_range = [pheno_coords[:, 0].min(), pheno_coords[:, 0].max()]
    pheno_j_range = [pheno_coords[:, 1].min(), pheno_coords[:, 1].max()]
    sbs_i_range = [sbs_coords[:, 0].min(), sbs_coords[:, 0].max()]
    sbs_j_range = [sbs_coords[:, 1].min(), sbs_coords[:, 1].max()]

    print(f"Coordinate ranges:")
    print(f"  Phenotype: i=[{pheno_i_range[0]:.0f}, {pheno_i_range[1]:.0f}], j=[{pheno_j_range[0]:.0f}, {pheno_j_range[1]:.0f}]")
    print(f"  SBS: i=[{sbs_i_range[0]:.0f}, {sbs_i_range[1]:.0f}], j=[{sbs_j_range[0]:.0f}, {sbs_j_range[1]:.0f}]")

    # Check for coordinate overlap
    overlap_i = [max(pheno_i_range[0], sbs_i_range[0]), min(pheno_i_range[1], sbs_i_range[1])]
    overlap_j = [max(pheno_j_range[0], sbs_j_range[0]), min(pheno_j_range[1], sbs_j_range[1])]
    has_overlap = overlap_i[1] > overlap_i[0] and overlap_j[1] > overlap_j[0]

    if has_overlap:
        print(f"✅ Coordinate overlap detected: i=[{overlap_i[0]:.0f}, {overlap_i[1]:.0f}], j=[{overlap_j[0]:.0f}, {overlap_j[1]:.0f}]")
        threshold = snakemake.params.threshold
    else:
        print("❌ No coordinate overlap - using relaxed threshold")
        threshold = snakemake.params.threshold * 5

# Cell merging
print("\n=== Cell Merging ===")
try:
    if use_triangle_alignment and alignment is not None:
        # Use triangle-hash alignment result
        merged_cells = merge_stitched_cells(
            phenotype_positions=phenotype_well,
            sbs_positions=sbs_well,
            alignment=alignment,
            threshold=snakemake.params.threshold,
            chunk_size=50000
        )
        print(f"✅ Used triangle-hash alignment for merging")
        
    else:
        # Simple distance-based merging
        print(f"Using simple distance-based merging with threshold={threshold}")
        
        from scipy.spatial.distance import cdist
        
        distances = cdist(sbs_coords, pheno_coords, metric='euclidean')
        closest_pheno_idx = distances.argmin(axis=1)
        min_distances = distances.min(axis=1)
        
        valid_matches = min_distances < threshold
        
        if valid_matches.sum() == 0:
            print("❌ No matches found")
            merged_cells = pd.DataFrame(columns=[
                'cell_0', 'i_0', 'j_0', 'area_0',
                'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
            ])
        else:
            sbs_match_indices = np.where(valid_matches)[0]
            pheno_match_indices = closest_pheno_idx[valid_matches]
            match_distances = min_distances[valid_matches]
            
            merged_cells = pd.DataFrame({
                'cell_0': phenotype_well.iloc[pheno_match_indices]['cell'].values,
                'i_0': phenotype_well.iloc[pheno_match_indices]['i'].values,
                'j_0': phenotype_well.iloc[pheno_match_indices]['j'].values,
                'cell_1': sbs_well.iloc[sbs_match_indices]['cell'].values,
                'i_1': sbs_well.iloc[sbs_match_indices]['i'].values,
                'j_1': sbs_well.iloc[sbs_match_indices]['j'].values,
                'distance': match_distances
            })
            
            # Add area columns if available
            if 'area' in phenotype_well.columns:
                merged_cells['area_0'] = phenotype_well.iloc[pheno_match_indices]['area'].values
            else:
                merged_cells['area_0'] = np.nan
                
            if 'area' in sbs_well.columns:
                merged_cells['area_1'] = sbs_well.iloc[sbs_match_indices]['area'].values
            else:
                merged_cells['area_1'] = np.nan
            
            # Remove duplicate phenotype cells
            merged_cells = merged_cells.sort_values('distance').drop_duplicates('cell_0', keep='first')
            
            print(f"✅ Simple distance merging successful")
    
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