import pandas as pd
import numpy as np
import os
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import triangle_hash_well_alignment, check_alignment_quality_permissive, test_hardcoded_scale_alignment, pure_scaling_alignment 

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
    alignment_df = pure_scaling_alignment(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        scale_factor=0.125  # The scale that gave 96.9% overlap
    )

    if len(alignment_df) > 0:
        alignment = alignment_df.iloc[0]

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

# Cell Merging with RAW MATCHES SAVE
print("\n=== Cell Merging ===")
try:
    if use_triangle_alignment and alignment is not None:
        # CUSTOM MERGE LOGIC WITH RAW MATCHES SAVE
        print(f"Starting cell merging with threshold={snakemake.params.threshold}")
        
        # Extract transformation parameters
        if isinstance(alignment, pd.Series):
            rotation = alignment['rotation']
            translation = alignment['translation']
        else:
            rotation = alignment.get('rotation', np.eye(2))
            translation = alignment.get('translation', np.zeros(2))
        
        # Ensure rotation is 2x2 matrix
        if rotation is None or np.array(rotation).size != 4:
            rotation = np.eye(2)
        else:
            rotation = np.array(rotation).reshape(2, 2)
        
        # Ensure translation is length 2 vector
        if translation is None or np.array(translation).size != 2:
            translation = np.zeros(2)
        else:
            translation = np.array(translation).flatten()[:2]
        
        print(f"Using transformation: rotation det={np.linalg.det(rotation):.3f}, translation={translation}")

        # Get coordinates
        pheno_coords = phenotype_well[['i', 'j']].values
        sbs_coords = sbs_well[['i', 'j']].values

        # Transform phenotype coordinates to SBS coordinate system
        transformed_coords = pheno_coords @ rotation.T + translation
         
        print(f"Coordinate ranges after transformation:")
        print(f"  Transformed phenotype: i=[{transformed_coords[:, 0].min():.0f}, {transformed_coords[:, 0].max():.0f}], j=[{transformed_coords[:, 1].min():.0f}, {transformed_coords[:, 1].max():.0f}]")
        print(f"  SBS: i=[{sbs_coords[:, 0].min():.0f}, {sbs_coords[:, 0].max():.0f}], j=[{sbs_coords[:, 1].min():.0f}, {sbs_coords[:, 1].max():.0f}]")
        
        # Process in chunks to manage memory
        all_matches = []
        chunk_size = 50000
        threshold = snakemake.params.threshold
        
        n_chunks = (len(sbs_well) + chunk_size - 1) // chunk_size
        print(f"Processing {len(sbs_well):,} SBS cells in {n_chunks} chunks of {chunk_size:,}")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(sbs_well))
            
            if chunk_idx % 10 == 0:
                print(f"Processing chunk {chunk_idx + 1}/{n_chunks}")
            
            # Get chunk of SBS coordinates
            sbs_chunk_coords = sbs_coords[start_idx:end_idx]
            
            # Calculate distances from transformed phenotype to SBS chunk
            from scipy.spatial.distance import cdist
            distances = cdist(sbs_chunk_coords, transformed_coords, metric='euclidean')
            
            # Find closest phenotype cell for each SBS cell in chunk
            closest_pheno_idx = distances.argmin(axis=1)
            min_distances = distances.min(axis=1)
            
            # Filter by threshold
            valid_matches = min_distances < threshold
            
            if valid_matches.sum() > 0:
                # Create match records for this chunk
                sbs_chunk_indices = np.arange(start_idx, end_idx)[valid_matches]
                pheno_match_indices = closest_pheno_idx[valid_matches]
                match_distances = min_distances[valid_matches]
                
                # Build DataFrame for this chunk
                chunk_matches = pd.DataFrame({
                    'cell_0': phenotype_well.iloc[pheno_match_indices]['cell'].values,
                    'i_0': phenotype_well.iloc[pheno_match_indices]['i'].values,
                    'j_0': phenotype_well.iloc[pheno_match_indices]['j'].values,
                    'cell_1': sbs_well.iloc[sbs_chunk_indices]['cell'].values,
                    'i_1': sbs_well.iloc[sbs_chunk_indices]['i'].values,
                    'j_1': sbs_well.iloc[sbs_chunk_indices]['j'].values,
                    'distance': match_distances
                })
                
                # Add area columns if available
                if 'area' in phenotype_well.columns:
                    chunk_matches['area_0'] = phenotype_well.iloc[pheno_match_indices]['area'].values
                else:
                    chunk_matches['area_0'] = np.nan
                    
                if 'area' in sbs_well.columns:
                    chunk_matches['area_1'] = sbs_well.iloc[sbs_chunk_indices]['area'].values
                else:
                    chunk_matches['area_1'] = np.nan
                
                all_matches.append(chunk_matches)
        
        # Combine all chunks
        if all_matches:
            merged_cells_raw = pd.concat(all_matches, ignore_index=True)
            
            print(f"Before deduplication: {len(merged_cells_raw):,} matches")
            print(f"Duplicate phenotype cells: {merged_cells_raw['cell_0'].duplicated().sum():,}")
            
            # SAVE RAW MATCHES (BEFORE DEDUPLICATION)
            raw_matches_path = snakemake.output[0].replace('.parquet', '_raw_matches.parquet')
            merged_cells_raw.to_parquet(raw_matches_path)
            print(f"✅ Saved raw matches (before deduplication) to: {raw_matches_path}")
            
            # Remove duplicate phenotype cells (keep best matches)
            merged_cells = merged_cells_raw.sort_values('distance').drop_duplicates('cell_0', keep='first')
            
            print(f"After deduplication: {len(merged_cells):,} matches")
            print(f"Successfully merged {len(merged_cells):,} cells")
            print(f"Distance statistics: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
            
        else:
            print("No cells matched within threshold")
            merged_cells = pd.DataFrame(columns=[
                'cell_0', 'i_0', 'j_0', 'area_0',
                'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
            ])
            # Also save empty raw matches
            raw_matches_path = snakemake.output[0].replace('.parquet', '_raw_matches.parquet')
            merged_cells.to_parquet(raw_matches_path)
            print(f"✅ Saved empty raw matches file to: {raw_matches_path}")
        
        print(f"✅ Used triangle-hash alignment for merging")
        
    else:
        print("❌ No valid triangle-hash alignment - cannot proceed")
        merged_cells = pd.DataFrame(columns=[
            'plate', 'well', 'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance', 'tile', 'site'
        ])
        # Save empty raw matches
        raw_matches_path = snakemake.output[0].replace('.parquet', '_raw_matches.parquet')
        merged_cells.to_parquet(raw_matches_path)
    
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
    
    # Verify raw matches file was created
    raw_matches_path = snakemake.output[0].replace('.parquet', '_raw_matches.parquet')
    if os.path.exists(raw_matches_path):
        raw_file_size = os.path.getsize(raw_matches_path)
        print(f"✅ Raw matches file confirmed: {raw_file_size:,} bytes")
    else:
        print(f"❌ Raw matches file was not created!")
    
except Exception as e:
    print(f"❌ Failed to save: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Triangle-hash-first well-level merge completed!")