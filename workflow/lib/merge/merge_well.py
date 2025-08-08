"""
Corrected scale factor logic for merge_well.py

The key insight: SBS coordinates should be ~8x SMALLER than phenotype coordinates
due to 40x vs 10x magnification + 2x2 binning.
"""

def calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size):
    """
    Calculate expected scale factor with correct physics.
    
    For your setup:
    - Phenotype: 40x magnification, no binning, 0.1625 μm/pixel
    - SBS: 10x magnification, 2x2 binning, 1.3 μm/pixel
    
    Expected scale = SBS_coordinate_range / Phenotype_coordinate_range
    Should be ~1/8 = 0.125 because SBS covers same physical area with fewer pixels
    """
    if phenotype_pixel_size and sbs_pixel_size:
        # This gives us the ratio of physical distances per pixel
        # SBS pixels represent more physical distance, so coordinates are smaller
        expected_scale = phenotype_pixel_size / sbs_pixel_size
        return expected_scale
    return None


def corrected_scale_analysis(phenotype_positions, sbs_positions, 
                           phenotype_pixel_size, sbs_pixel_size):
    """
    Corrected scale analysis with proper physics understanding.
    """
    # Calculate coordinate ranges
    pheno_i_range = phenotype_positions['i'].max() - phenotype_positions['i'].min()
    pheno_j_range = phenotype_positions['j'].max() - phenotype_positions['j'].min()
    sbs_i_range = sbs_positions['i'].max() - sbs_positions['i'].min()
    sbs_j_range = sbs_positions['j'].max() - sbs_positions['j'].min()
    
    # Empirical scale (what we observe)
    empirical_scale_i = sbs_i_range / pheno_i_range if pheno_i_range > 0 else 1.0
    empirical_scale_j = sbs_j_range / pheno_j_range if pheno_j_range > 0 else 1.0
    empirical_scale = (empirical_scale_i + empirical_scale_j) / 2
    
    # Expected scale (what physics tells us)
    expected_scale = calculate_expected_scale_factor(phenotype_pixel_size, sbs_pixel_size)
    
    print(f"=== CORRECTED Scale Analysis ===")
    print(f"Coordinate ranges:")
    print(f"  Phenotype: i={pheno_i_range:.0f}, j={pheno_j_range:.0f}")
    print(f"  SBS: i={sbs_i_range:.0f}, j={sbs_j_range:.0f}")
    print(f"Empirical scale factor (SBS/phenotype): {empirical_scale:.3f}")
    
    if expected_scale:
        print(f"Expected scale factor: {expected_scale:.3f}")
        scale_difference = abs(empirical_scale - expected_scale)
        print(f"Scale difference: {scale_difference:.3f}")
        
        # Check if scales match (within 50% tolerance)
        if scale_difference < expected_scale * 0.5:
            print("✅ Scales match expected physics - coordinates are correct!")
            return {
                'scales_match': True,
                'empirical_scale': empirical_scale,
                'expected_scale': expected_scale,
                'scale_difference': scale_difference,
                'coordinate_correction_needed': False
            }
        else:
            print("⚠️ Scale mismatch detected")
            return {
                'scales_match': False,
                'empirical_scale': empirical_scale,
                'expected_scale': expected_scale,
                'scale_difference': scale_difference,
                'coordinate_correction_needed': True
            }
    else:
        print("No pixel size metadata available")
        return {
            'scales_match': None,
            'empirical_scale': empirical_scale,
            'expected_scale': None,
            'coordinate_correction_needed': False
        }


def analyze_overlap_with_correct_scale(phenotype_positions, sbs_positions, alignment):
    """
    Analyze coordinate overlap with correct scale understanding.
    """
    print(f"\n=== Overlap Analysis with Correct Scale ===")
    
    # Apply the transformation to phenotype coordinates
    rotation = np.array([alignment['rotation_1'], alignment['rotation_2']])
    translation = alignment['translation']
    
    # Transform phenotype coordinates to SBS coordinate system
    pheno_coords = phenotype_positions[['i', 'j']].values
    transformed_coords = pheno_coords @ rotation.T + translation
    
    # Find overlap region in SBS coordinate system
    sbs_coords = sbs_positions[['i', 'j']].values
    
    # Calculate bounds
    sbs_i_min, sbs_i_max = sbs_coords[:, 0].min(), sbs_coords[:, 0].max()
    sbs_j_min, sbs_j_max = sbs_coords[:, 1].min(), sbs_coords[:, 1].max()
    
    transformed_i_min, transformed_i_max = transformed_coords[:, 0].min(), transformed_coords[:, 0].max()
    transformed_j_min, transformed_j_max = transformed_coords[:, 1].min(), transformed_coords[:, 1].max()
    
    # Find intersection
    overlap_i_min = max(sbs_i_min, transformed_i_min)
    overlap_i_max = min(sbs_i_max, transformed_i_max)
    overlap_j_min = max(sbs_j_min, transformed_j_min)
    overlap_j_max = min(sbs_j_max, transformed_j_max)
    
    print(f"Coordinate bounds (in SBS coordinate system):")
    print(f"  SBS: i=[{sbs_i_min:.0f}, {sbs_i_max:.0f}], j=[{sbs_j_min:.0f}, {sbs_j_max:.0f}]")
    print(f"  Transformed phenotype: i=[{transformed_i_min:.0f}, {transformed_i_max:.0f}], j=[{transformed_j_min:.0f}, {transformed_j_max:.0f}]")
    print(f"  Overlap region: i=[{overlap_i_min:.0f}, {overlap_i_max:.0f}], j=[{overlap_j_min:.0f}, {overlap_j_max:.0f}]")
    
    # Check if there's meaningful overlap
    if overlap_i_max > overlap_i_min and overlap_j_max > overlap_j_min:
        overlap_area = (overlap_i_max - overlap_i_min) * (overlap_j_max - overlap_j_min)
        sbs_total_area = (sbs_i_max - sbs_i_min) * (sbs_j_max - sbs_j_min)
        overlap_fraction = overlap_area / sbs_total_area if sbs_total_area > 0 else 0
        
        print(f"  Overlap area: {overlap_area:.0f} square pixels ({overlap_fraction:.1%} of SBS area)")
        
        # Count cells in overlap region
        sbs_in_overlap = np.sum(
            (sbs_coords[:, 0] >= overlap_i_min) & (sbs_coords[:, 0] <= overlap_i_max) &
            (sbs_coords[:, 1] >= overlap_j_min) & (sbs_coords[:, 1] <= overlap_j_max)
        )
        
        transformed_in_overlap = np.sum(
            (transformed_coords[:, 0] >= overlap_i_min) & (transformed_coords[:, 0] <= overlap_i_max) &
            (transformed_coords[:, 1] >= overlap_j_min) & (transformed_coords[:, 1] <= overlap_j_max)
        )
        
        print(f"Cells in overlap region:")
        print(f"  SBS: {sbs_in_overlap:,} ({100*sbs_in_overlap/len(sbs_positions):.1f}%)")
        print(f"  Phenotype: {transformed_in_overlap:,} ({100*transformed_in_overlap/len(phenotype_positions):.1f}%)")
        print(f"  Expected max matches: {min(sbs_in_overlap, transformed_in_overlap):,}")
        
        return {
            'has_overlap': True,
            'overlap_fraction': overlap_fraction,
            'sbs_cells_in_overlap': sbs_in_overlap,
            'phenotype_cells_in_overlap': transformed_in_overlap,
            'expected_max_matches': min(sbs_in_overlap, transformed_in_overlap)
        }
    else:
        print("❌ No meaningful overlap detected!")
        return {
            'has_overlap': False,
            'overlap_fraction': 0,
            'sbs_cells_in_overlap': 0,
            'phenotype_cells_in_overlap': 0,
            'expected_max_matches': 0
        }


def enhanced_alignment_with_correct_scale(
    phenotype_positions: pd.DataFrame,
    sbs_positions: pd.DataFrame,
    phenotype_pixel_size: float = None,
    sbs_pixel_size: float = None,
    **kwargs
) -> pd.DataFrame:
    """
    Enhanced alignment that understands the correct scale relationship.
    """
    print("=== Enhanced Alignment with Correct Scale Physics ===")
    
    # Step 1: Verify scale is as expected
    scale_analysis = corrected_scale_analysis(
        phenotype_positions, sbs_positions,
        phenotype_pixel_size, sbs_pixel_size
    )
    
    # Step 2: Run alignment (coordinates should already be correct)
    result = stitched_well_alignment(
        phenotype_positions, sbs_positions,
        phenotype_pixel_size=phenotype_pixel_size,
        sbs_pixel_size=sbs_pixel_size,
        **kwargs
    )
    
    # Step 3: Analyze overlap with the result
    if not result.empty and len(result) > 0:
        alignment = result.iloc[0]
        overlap_analysis = analyze_overlap_with_correct_scale(
            phenotype_positions, sbs_positions, alignment
        )
        
        # Add overlap analysis to result
        result = result.copy()
        for key, value in overlap_analysis.items():
            result[f'overlap_{key}'] = value
    
    return result