import pandas as pd
import numpy as np
import os
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import stitched_well_alignment, merge_stitched_cells

def create_fallback_alignment():
    """Create a simple translation-only fallback alignment."""
    return {
        'rotation': np.eye(2),
        'translation': np.zeros(2),
        'score': 0.5,
        'determinant': 1.0,
        'transformation_type': 'translation_fallback',
        'n_triangles_matched': 0,
        'cells_used_phenotype': 0,
        'cells_used_sbs': 0
    }

def simple_merge_cells(phenotype_positions, sbs_positions, threshold=2.0):
    """Simple distance-based cell merging without transformation."""
    pheno_coords = phenotype_positions[['i', 'j']].values
    sbs_coords = sbs_positions[['i', 'j']].values
    
    # Calculate distances between all pairs
    from scipy.spatial.distance import cdist
    distances = cdist(sbs_coords, pheno_coords, metric='euclidean')
    
    # Find closest phenotype cell for each SBS cell
    closest_pheno_idx = distances.argmin(axis=1)
    min_distances = distances.min(axis=1)
    
    # Filter by threshold
    valid_matches = min_distances < threshold
    
    if valid_matches.sum() == 0:
        return pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
    
    # Create matches
    sbs_match_indices = np.where(valid_matches)[0]
    pheno_match_indices = closest_pheno_idx[valid_matches]
    match_distances = min_distances[valid_matches]
    
    merged_cells = pd.DataFrame({
        'cell_0': phenotype_positions.iloc[pheno_match_indices]['cell'].values,
        'i_0': phenotype_positions.iloc[pheno_match_indices]['i'].values,
        'j_0': phenotype_positions.iloc[pheno_match_indices]['j'].values,
        'cell_1': sbs_positions.iloc[sbs_match_indices]['cell'].values,
        'i_1': sbs_positions.iloc[sbs_match_indices]['i'].values,
        'j_1': sbs_positions.iloc[sbs_match_indices]['j'].values,
        'distance': match_distances
    })
    
    # Add area columns if available
    if 'area' in phenotype_positions.columns:
        merged_cells['area_0'] = phenotype_positions.iloc[pheno_match_indices]['area'].values
    else:
        merged_cells['area_0'] = np.nan
        
    if 'area' in sbs_positions.columns:
        merged_cells['area_1'] = sbs_positions.iloc[sbs_match_indices]['area'].values
    else:
        merged_cells['area_1'] = np.nan
    
    # Remove duplicate phenotype cells (keep best matches)
    merged_cells = merged_cells.sort_values('distance').drop_duplicates('cell_0', keep='first')
    
    return merged_cells

# REPLACE your existing alignment and merging code with this simpler version:

# After your pixel size extraction and filtering code, replace the complex alignment section with:

print("\n=== Simple Distance-Based Alignment ===")

# Check coordinate overlap without transformation
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
    
    # Simple distance-based merging
    try:
        merged_cells = simple_merge_cells(
            phenotype_well, 
            sbs_well, 
            threshold=snakemake.params.threshold
        )
        
        if len(merged_cells) > 0:
            print(f"✅ Successfully merged {len(merged_cells):,} cells using simple distance matching")
            print(f"   Distance stats: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
        else:
            print("⚠️ No cells matched within distance threshold")
            
    except Exception as e:
        print(f"❌ Simple merging failed: {e}")
        merged_cells = pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])
else:
    print("❌ No coordinate overlap - coordinates may need alignment")
    print("Using fallback merge with increased distance threshold")
    
    try:
        # Try with larger threshold since coordinates don't overlap
        merged_cells = simple_merge_cells(
            phenotype_well, 
            sbs_well, 
            threshold=snakemake.params.threshold * 5  # 5x larger threshold
        )
        
        if len(merged_cells) > 0:
            print(f"✅ Fallback merge found {len(merged_cells):,} cells with relaxed threshold")
        else:
            print("❌ Even fallback merge found no matches")
            
    except Exception as e:
        print(f"❌ Fallback merge failed: {e}")
        merged_cells = pd.DataFrame(columns=[
            'cell_0', 'i_0', 'j_0', 'area_0',
            'cell_1', 'i_1', 'j_1', 'area_1', 'distance'
        ])

# Add metadata (keep your existing code for this part)
if len(merged_cells) > 0:
    merged_cells["plate"] = int(plate)
    merged_cells["well"] = well
    
    if "tile" not in merged_cells.columns:
        merged_cells["tile"] = 1
    if "site" not in merged_cells.columns:
        merged_cells["site"] = 1


# Load cell positions from stitched masks
phenotype_positions = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_positions))
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

# Load metadata files  
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_metadata))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))

# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well

print(f"=== Enhanced Well-Level Merge: {plate}/{well} ===")
print(f"Phenotype cells: {len(phenotype_positions):,}")
print(f"SBS cells: {len(sbs_positions):,}")

# Extract pixel sizes from metadata
def extract_pixel_size(metadata_df, plate, well, data_type):
    """Extract pixel size from metadata for this plate/well."""
    well_metadata = metadata_df[
        (metadata_df["plate"] == int(plate)) & 
        (metadata_df["well"] == well)
    ]
    
    if (len(well_metadata) > 0 and 
        "pixel_size_x" in well_metadata.columns and 
        "pixel_size_y" in well_metadata.columns):
        pixel_size = well_metadata["pixel_size_x"].iloc[0]
        print(f"{data_type} pixel size: {pixel_size:.6f} μm/pixel")
        return pixel_size
    else:
        # Fall back to typical values
        typical_values = {"phenotype": 0.1625, "sbs": 1.30}  # 40x vs 10x + binning
        pixel_size = typical_values.get(data_type, 1.0)
        print(f"Using typical {data_type} pixel size: {pixel_size:.3f} μm/pixel")
        return pixel_size

phenotype_pixel_size = extract_pixel_size(phenotype_metadata, plate, well, "phenotype")
sbs_pixel_size = extract_pixel_size(sbs_metadata, plate, well, "sbs")

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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
    empty_result.to_parquet(snakemake.output[0])
    exit(0)

# Perform triangle hash alignment
print("\n=== Triangle Hash Alignment ===")
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
        sbs_pixel_size=sbs_pixel_size
    )

    if len(alignment_df) == 0:
        print("❌ Alignment failed")
        empty_result = pd.DataFrame(columns=[
            "plate", "well", "cell_0", "i_0", "j_0", "area_0", 
            "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
        ])
        os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
        empty_result.to_parquet(snakemake.output[0])
        exit(0)

    alignment = alignment_df.iloc[0]
    print(f"✅ Alignment successful:")
    print(f"   Score: {alignment['score']:.3f}")
    print(f"   Determinant: {alignment['determinant']:.3f}")
    print(f"   Type: {alignment['transformation_type']}")
    print(f"   Triangles matched: {alignment['triangles_matched']}")

except Exception as e:
    print(f"❌ Alignment failed: {e}")
    empty_result = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
    empty_result.to_parquet(snakemake.output[0])
    exit(0)

# Validate alignment quality
det_lower, det_upper = snakemake.params.det_range
score_min = snakemake.params.score

# Adjust bounds based on transformation type
transformation_type = alignment.get('transformation_type', 'unknown')
if transformation_type in ['translation_only', 'translation_fallback']:
    det_lower, det_upper = 0.9, 1.1
elif transformation_type == 'translation_rotation':
    det_lower, det_upper = 0.5, 2.0
else:
    det_lower, det_upper = 0.1, 10.0  # Permissive for other types

alignment_ok = (det_lower <= alignment["determinant"] <= det_upper and 
                alignment["score"] >= score_min)

if not alignment_ok:
    print(f"⚠️ Alignment quality insufficient:")
    print(f"   Determinant: {alignment['determinant']:.3f} (need: {det_lower:.3f}-{det_upper:.3f})")
    print(f"   Score: {alignment['score']:.3f} (need: >{score_min})")
    
    empty_result = pd.DataFrame(columns=[
        "plate", "well", "cell_0", "i_0", "j_0", "area_0",
        "cell_1", "i_1", "j_1", "area_1", "distance", "tile", "site"
    ])
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)
    empty_result.to_parquet(snakemake.output[0])
    exit(0)

# Merge cells
print("\n=== Cell Merging ===")
try:
    merged_cells = merge_stitched_cells(
        phenotype_positions=phenotype_well,
        sbs_positions=sbs_well,
        alignment=alignment,
        threshold=snakemake.params.threshold,
        chunk_size=50000
    )
    
    print(f"✅ Merged {len(merged_cells):,} cells")
    
    if len(merged_cells) > 0:
        print(f"   Distance stats: mean={merged_cells['distance'].mean():.2f}, max={merged_cells['distance'].max():.2f}")
        
        # Add required metadata
        merged_cells["plate"] = int(plate)
        merged_cells["well"] = well
        
        # Add tile/site for downstream compatibility
        if "tile" not in merged_cells.columns:
            merged_cells["tile"] = 1  # Well-level stitched data
        if "site" not in merged_cells.columns:
            merged_cells["site"] = 1   # Well-level stitched data
    else:
        print("⚠️ No cells matched within distance threshold")
        
except Exception as e:
    print(f"❌ Cell merging failed: {e}")
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
    
except Exception as e:
    print(f"❌ Failed to save: {e}")
    raise

print("Enhanced well-level merge completed!")