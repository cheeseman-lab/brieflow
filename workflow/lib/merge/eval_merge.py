"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import yaml

from lib.shared.eval import plot_plate_heatmap
from lib.shared.configuration_utils import plot_merge_example
from lib.merge.merge import build_linear_model
from lib.merge.well_alignment import (
    sample_region_for_alignment,
    calculate_scale_factor_from_positions,
    scale_coordinates,
)

def display_matched_and_unmatched_cells_for_site(root_fp, plate, well, selected_site=None, 
                                               distance_threshold=15.0, max_display_rows=1000, verbose=False):
    """Display matched and unmatched cells using stitched_cell_id for matching.
    
    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier  
        well (str): Well identifier
        selected_site (str, optional): Site to filter by (if None, shows first available site)
        distance_threshold (float): Maximum distance to show matches
        max_display_rows (int): Maximum rows to display for performance
    """
    from pathlib import Path
    import pandas as pd
    
    # Construct paths to required files
    root_path = Path(root_fp)
    merge_fp = root_path / "merge"
    
    merged_cells_path = merge_fp / "well_cell_merge" / f"P-{plate}_W-{well}__raw_matches.parquet"
    phenotype_transformed_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    sbs_positions_path = merge_fp / "cell_positions" / f"P-{plate}_W-{well}__sbs_cell_positions.parquet"
    
    # Check if all required files exist
    missing_files = []
    if not merged_cells_path.exists():
        missing_files.append(f"Raw matches: {merged_cells_path}")
    if not phenotype_transformed_path.exists():
        missing_files.append(f"Transformed phenotype: {phenotype_transformed_path}")
    if not sbs_positions_path.exists():
        missing_files.append(f"SBS positions: {sbs_positions_path}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        return None
    
    try:
        # Load all datasets
        if verbose:
            print(f"📁 Loading cell matching data for Plate {plate}, Well {well}...")
        merged_df = pd.read_parquet(merged_cells_path)
        phenotype_transformed = pd.read_parquet(phenotype_transformed_path)
        sbs_positions = pd.read_parquet(sbs_positions_path)
        
        print(f"✅ {len(merged_df)} total cell matches")
        print(f"✅ {len(phenotype_transformed)} transformed phenotype cells")
        print(f"✅ {len(sbs_positions)} SBS cells")
        
        # Get available sites from merged data
        available_sites = sorted(merged_df['site'].unique()) if 'site' in merged_df.columns else []
        if verbose:
            print(f"📍 Available sites: {available_sites}")
        
        # Select site to display
        if selected_site is None:
            if available_sites:
                selected_site = available_sites[0]
                print(f"🎯 Auto-selected site: {selected_site}")
            else:
                print("❌ No sites found in merged data")
                return None
        elif selected_site not in available_sites:
            print(f"❌ Selected site '{selected_site}' not found. Available: {available_sites}")
            return None
        else:
            print(f"🎯 Using selected site: {selected_site}")
        
        # Filter merged data by site and distance threshold
        site_merged = merged_df[merged_df['site'] == selected_site].copy()
        filtered_merged = site_merged[site_merged['distance'] <= distance_threshold].copy()
        
        # Filter position datasets based on spatial coordinates
        
        # For SBS positions, filter by site
        if 'site' in sbs_positions.columns:
            site_sbs = sbs_positions[sbs_positions['site'] == selected_site].copy()
        elif 'tile' in sbs_positions.columns:
            # Sometimes SBS data uses 'tile' instead of 'site'
            site_sbs = sbs_positions[sbs_positions['tile'] == selected_site].copy()
        else:
            print("⚠️  No 'site' or 'tile' column in sbs_positions, using all cells")
            site_sbs = sbs_positions.copy()
        
        # Get the coordinate range of SBS cells in the selected site
        if len(site_sbs) > 0:
            sbs_i_min, sbs_i_max = site_sbs['i'].min(), site_sbs['i'].max()
            sbs_j_min, sbs_j_max = site_sbs['j'].min(), site_sbs['j'].max()
            if verbose:
                print(f"   SBS coordinate range: i=[{sbs_i_min:.1f}, {sbs_i_max:.1f}], j=[{sbs_j_min:.1f}, {sbs_j_max:.1f}]")
            
            # For phenotype_transformed, filter by coordinates within SBS range
            site_phenotype = phenotype_transformed[
                (phenotype_transformed['i'] >= sbs_i_min) & 
                (phenotype_transformed['i'] <= sbs_i_max) &
                (phenotype_transformed['j'] >= sbs_j_min) & 
                (phenotype_transformed['j'] <= sbs_j_max)
            ].copy()

        else:
            print("❌ No SBS cells found for selected site")
            return None
        
        print(f"🔍 Site '{selected_site}' cell counts:")
        print(f"   Merged matches: {len(site_merged)}")
        print(f"   Matches within {distance_threshold}px: {len(filtered_merged)}")
        print(f"   Total phenotype cells at site '{selected_site}': {len(site_phenotype)}")
        print(f"   Total SBS cells at site '{selected_site}': {len(site_sbs)}")
        

        
        # For phenotype: match using stitched_cell_id_0 from raw_matches vs stitched_cell_id from phenotype_transformed
        if 'stitched_cell_id_0' not in filtered_merged.columns:
            print("❌ Error: 'stitched_cell_id_0' column not found in raw_matches data")
            return None
        
        if 'stitched_cell_id' not in site_phenotype.columns:
            print("❌ Error: 'stitched_cell_id' column not found in phenotype_transformed data")
            return None
        
        matched_phenotype_stitched_ids = set(filtered_merged['stitched_cell_id_0'].dropna().unique())
        unmatched_phenotype = site_phenotype[~site_phenotype['stitched_cell_id'].isin(matched_phenotype_stitched_ids)].copy()
        
        # For SBS: match using stitched_cell_id_1 from raw_matches vs stitched_cell_id from sbs_positions
        if 'stitched_cell_id_1' not in filtered_merged.columns:
            print("❌ Error: 'stitched_cell_id_1' column not found in raw_matches data")
            return None
            
        if 'stitched_cell_id' not in site_sbs.columns:
            print("❌ Error: 'stitched_cell_id' column not found in sbs_positions data")
            return None
            
        matched_sbs_stitched_ids = set(filtered_merged['stitched_cell_id_1'].dropna().unique())
        unmatched_sbs = site_sbs[~site_sbs['stitched_cell_id'].isin(matched_sbs_stitched_ids)].copy()
        
        print(f"   Unmatched phenotype cells: {len(unmatched_phenotype)}")
        print(f"   Unmatched SBS cells: {len(unmatched_sbs)}")
        
        # Calculate match rates
        total_phenotype = len(site_phenotype)
        total_sbs = len(site_sbs)
        match_rate_phenotype = len(filtered_merged) / total_phenotype if total_phenotype > 0 else 0
        match_rate_sbs = len(filtered_merged) / total_sbs if total_sbs > 0 else 0
        
        print(f"   Phenotype match rate at site: {match_rate_phenotype:.1%}")
        print(f"   SBS match rate at site: {match_rate_sbs:.1%}")
        
        if len(filtered_merged) == 0:
            print(f"⚠️  No matches found within {distance_threshold}px threshold")
        
        # Calculate statistics for matched cells
        if len(filtered_merged) > 0:
            distances = filtered_merged['distance']
            print(f"   Distance statistics for matched cells:")
            print(f"     Mean: {distances.mean():.2f}px")
            print(f"     Median: {distances.median():.2f}px")
            print(f"     Min: {distances.min():.2f}px")
            print(f"     Max: {distances.max():.2f}px")
            print(f"     Within 5px: {(distances <= 5).sum()} ({(distances <= 5).sum()/len(distances)*100:.1f}%)")
            print(f"     Within 10px: {(distances <= 10).sum()} ({(distances <= 10).sum()/len(distances)*100:.1f}%)")
        
        if verbose:
            # Prepare display data - limit rows for performance
            display_sections = []
            
            # 1. Matched cells
            if len(filtered_merged) > 0:
                display_matched = filtered_merged.head(max_display_rows // 3) if len(filtered_merged) > max_display_rows // 3 else filtered_merged
                display_matched = display_matched.assign(match_status='MATCHED').copy()
                display_sections.append(('MATCHED CELLS', display_matched, len(filtered_merged)))
            
            # 2. Unmatched phenotype cells
            if len(site_phenotype) > 0:
                display_unmatched_ph = site_phenotype.head(max_display_rows // 3) if len(site_phenotype) > max_display_rows // 3 else site_phenotype
                # Standardize columns to match merged data format
                unmatched_ph_display = pd.DataFrame({
                    'plate': plate,
                    'well': well,
                    'site': selected_site,
                    'tile': selected_site,
                    'cell_0': display_unmatched_ph.get('cell', pd.NA),
                    'cell_1': pd.NA,
                    'i_0': display_unmatched_ph['i'],
                    'j_0': display_unmatched_ph['j'],
                    'i_1': pd.NA,
                    'j_1': pd.NA,
                    'area_0': display_unmatched_ph.get('area', pd.NA),
                    'area_1': pd.NA,
                    'distance': pd.NA,
                    'stitched_cell_id_0': display_unmatched_ph['stitched_cell_id'],
                    'stitched_cell_id_1': pd.NA,
                    'match_status': 'RAW_PHENOTYPE'
                })
                display_sections.append(('RAW PHENOTYPE CELLS', unmatched_ph_display, len(site_phenotype)))
            
            # 3. Unmatched SBS cells
            if len(site_sbs) > 0:
                display_unmatched_sbs = site_sbs.head(max_display_rows // 3) if len(site_sbs) > max_display_rows // 3 else site_sbs
                # Standardize columns to match merged data format
                unmatched_sbs_display = pd.DataFrame({
                    'plate': plate,
                    'well': well,
                    'site': selected_site,
                    'tile': selected_site,
                    'cell_0': pd.NA,
                    'cell_1': display_unmatched_sbs.get('cell', pd.NA),
                    'i_0': pd.NA,
                    'j_0': pd.NA,
                    'i_1': display_unmatched_sbs['i'],
                    'j_1': display_unmatched_sbs['j'],
                    'area_0': pd.NA,
                    'area_1': display_unmatched_sbs.get('area', pd.NA),
                    'distance': pd.NA,
                    'stitched_cell_id_0': pd.NA,
                    'stitched_cell_id_1': display_unmatched_sbs['stitched_cell_id'],
                    'match_status': 'RAW_SBS'
                })
                display_sections.append(('RAW SBS CELLS', unmatched_sbs_display, len(unmatched_sbs)))
            
            # Display each section
            display_columns = [
                'match_status', 'stitched_cell_id_0', 'stitched_cell_id_1', 'cell_0', 'cell_1', 
                'i_0', 'j_0', 'i_1', 'j_1', 'area_0', 'area_1', 'distance'
            ]
            
            # Set pandas display options for better formatting
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 15)
            
            print("\n" + "="*160)
            print(f"CELL MATCHING ANALYSIS - SITE: {selected_site}")
            print(f"Distance Threshold: ≤{distance_threshold}px | Matching by stitched_cell_id")
            print("="*160)
        
            for section_name, section_data, total_count in display_sections:
                print(f"\n{section_name} (Showing {len(section_data)} of {total_count}):")
                print("-" * 120)
                
                # Round numerical columns for better display
                display_data = section_data.copy()
                numerical_cols = ['i_0', 'j_0', 'i_1', 'j_1', 'area_0', 'area_1', 'distance']
                for col in numerical_cols:
                    if col in display_data.columns:
                        if col == 'distance':
                            display_data[col] = pd.to_numeric(display_data[col], errors='coerce').round(2)
                        else:
                            display_data[col] = pd.to_numeric(display_data[col], errors='coerce').round(1)
                
                # Show only existing columns
                existing_display_cols = [col for col in display_columns if col in display_data.columns]
                print(display_data[existing_display_cols].to_string(index=False))
            
            # Reset pandas options
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')  
            pd.reset_option('display.max_colwidth')
            
            print("\n" + "="*160)
        
        # Create enhanced visualization
        create_enhanced_match_visualization(
            matched_data=filtered_merged,
            site_phenotype=site_phenotype,
            site_sbs=site_sbs,
            unmatched_phenotype=unmatched_phenotype,
            unmatched_sbs=unmatched_sbs,
            site=selected_site,
            distance_threshold=distance_threshold
        )
        
    except Exception as e:
        print(f"❌ Error loading or processing cell data: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_enhanced_match_visualization(matched_data, site_phenotype, site_sbs,
                                        unmatched_phenotype, unmatched_sbs,
                                        site, distance_threshold):
    """Create enhanced visualization showing matched and unmatched cells.
    
    Args:
        matched_data (pd.DataFrame): Matched cell data
        site_phenotype (pd.DataFrame): Raw phenotype cells
        site_sbs (pd.DataFrame): Raw SBS cells
        unmatched_phenotype (pd.DataFrame): Unmatched phenotype cells
        unmatched_sbs (pd.DataFrame): Unmatched SBS cells
        site (str): Site name for title
        distance_threshold (float): Distance threshold used
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Cell Matching Analysis - Site: {site}', fontsize=16)
    
    # 1. Distance distribution histogram
    ax1 = axes[0, 0]
    if len(matched_data) > 0:
        distances = matched_data['distance']
        ax1.hist(distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(distance_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold: {distance_threshold}px')
        ax1.axvline(distances.mean(), color='orange', linestyle='-', linewidth=2, 
                    label=f'Mean: {distances.mean():.1f}px')
        ax1.set_xlabel('Distance (pixels)')
        ax1.set_ylabel('Count')
        ax1.set_title('Match Distance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No matches found', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Match Distance Distribution')
    
    # 2. Spatial overview - all cells
    ax2 = axes[0, 1]
    
    # Plot unmatched cells first (so matched cells appear on top)
    if len(site_phenotype) > 0:
        sample_size = min(2000, len(site_phenotype))  # Limit for performance
        sample_ph = site_phenotype.sample(n=sample_size) if len(site_phenotype) > sample_size else site_phenotype
        ax2.scatter(sample_ph['j'], sample_ph['i'], c='lightcoral', s=8, alpha=0.4, 
                   label=f'Raw Phenotype ({len(site_phenotype)})')
    
    if len(site_sbs) > 0:
        sample_size = min(2000, len(site_sbs))
        sample_sbs = site_sbs.sample(n=sample_size) if len(site_sbs) > sample_size else site_sbs
        ax2.scatter(sample_sbs['j'], sample_sbs['i'], c='lightblue', s=8, alpha=0.4, 
                   label=f'Raw SBS ({len(site_sbs)})')
    
    # Plot matched cells on top with borders
    if len(matched_data) > 0:
        sample_size = min(2000, len(matched_data))
        sample_matched = matched_data.sample(n=sample_size) if len(matched_data) > sample_size else matched_data
        # Color by distance with black borders for visibility
        scatter = ax2.scatter(sample_matched['j_0'], sample_matched['i_0'], 
                             c=sample_matched['distance'], s=15, alpha=0.9, 
                             cmap='viridis', edgecolors='black', linewidths=0.5,
                             label=f'Matched ({len(matched_data)})')
        plt.colorbar(scatter, ax=ax2, label='Match Distance (px)', shrink=0.8)
    
    ax2.set_xlabel('j (pixels)')
    ax2.set_ylabel('i (pixels)')
    ax2.set_title('Cell Positions Overview')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.invert_yaxis()  # Match image coordinates
    
    # 3. Match quality pie chart
    ax3 = axes[1, 0]
    
    total_phenotype = len(site_phenotype)
    total_sbs = len(site_sbs)
    
    # Show phenotype matching breakdown
    labels = ['Matched', 'Unmatched']
    sizes = [len(matched_data), len(site_phenotype)]
    colors = ['#2ecc71', '#e74c3c']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax3.set_title(f'Phenotype Match Rate\n({total_phenotype} total cells)')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Phenotype', 'SBS'],
        ['Total Cells', f'{total_phenotype}', f'{total_sbs}'],
        ['Matched Cells', f'{len(matched_data)}', f'{len(matched_data)}'],
        ['Unmatched Cells', f'{len(site_phenotype)}', f'{len(site_sbs)}'],
        ['Match Rate', f'{len(matched_data)/total_phenotype:.1%}' if total_phenotype > 0 else 'N/A',
         f'{len(matched_data)/total_sbs:.1%}' if total_sbs > 0 else 'N/A']
    ]
    
    if len(matched_data) > 0:
        distances = matched_data['distance']
        summary_data.extend([
            ['Mean Distance', f'{distances.mean():.1f}px', ''],
            ['Median Distance', f'{distances.median():.1f}px', ''],
            ['Excellent (≤2px)', f'{(distances <= 2).sum()}', ''],
            ['Very Good (≤5px)', f'{(distances <= 5).sum()}', ''],
            ['Good (≤10px)', f'{(distances <= 10).sum()}', ''],
            ['Fair (≤15px)', f'{(distances > 10).sum()}', ''],
        ])
    
    # Create table
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.show()


def run_enhanced_cell_matching_qc(root_fp, plate, well, selected_site=None, 
                                 distance_threshold=15.0, max_display_rows=1000, verbose=False):
    """Streamlined function focused only on cell matching analysis.
    
    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
        selected_site (str, optional): Specific site to display cells for
        distance_threshold (float): Maximum distance to show matches (default 15.0)
        max_display_rows (int): Maximum number of rows to display (default 1000)
    """
    print(f"🔬 CELL MATCHING ANALYSIS for Plate {plate}, Well {well}")
    print("="*80)
    
    # Run the cell matching analysis
    summary_stats = display_matched_and_unmatched_cells_for_site(
        root_fp, plate, well, selected_site, distance_threshold, max_display_rows, verbose
    )
    
    if summary_stats:
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Site: {summary_stats['site']}")
        print(f"   Phenotype cells: {summary_stats['total_phenotype_cells']:,}")
        print(f"   SBS cells: {summary_stats['total_sbs_cells']:,}")
        print(f"   Matched cells: {summary_stats['matched_cells']:,}")
        print(f"   Phenotype match rate: {summary_stats['phenotype_match_rate']:.1%}")
        print(f"   SBS match rate: {summary_stats['sbs_match_rate']:.1%}")
        if summary_stats['mean_match_distance']:
            print(f"   Mean match distance: {summary_stats['mean_match_distance']:.1f}px")
    
    return summary_stats


def run_well_alignment_qc(root_fp, plate, well, det_range, score, threshold, 
                          selected_site=None, distance_threshold=15.0, max_display_rows=1000):
    """Run complete QC visualization for a well alignment with merged cells display.
    
    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
        det_range (tuple): Determinant range from config
        score (float): Score threshold from config
        threshold (float): Distance threshold from config
        selected_site (str, optional): Specific site to display merged cells for
        distance_threshold (float): Maximum distance to show matches (default 15.0)
        max_display_rows (int): Maximum number of rows to display (default 1000)
    """
    print(f"Running Well Alignment QC for Plate {plate}, Well {well}")
    print("-" * 60)
    
    # Load alignment data
    alignment_data = load_well_alignment_outputs(root_fp, plate, well)
    
    # Display summary
    display_well_alignment_summary(alignment_data)
    
    return alignment_data

def load_well_alignment_outputs(root_fp, plate, well, verbose=False):
    """Load all alignment outputs for a specific well.
    
    Args:
        root_fp (str/Path): Root analysis directory path
        plate (str): Plate identifier
        well (str): Well identifier
        
    Returns:
        dict: Dictionary containing all loaded alignment data
    """
    root_fp = Path(root_fp)
    merge_fp = root_fp / "merge"
    
    outputs = {}
    
    # Load alignment parameters
    alignment_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__alignment.parquet"
    if alignment_path.exists():
        outputs['alignment_params'] = pd.read_parquet(alignment_path)
        if verbose:
            print(f"✅ Loaded alignment parameters: {len(outputs['alignment_params'])} entries")
    else:
        raise FileNotFoundError(f"Alignment parameters not found: {alignment_path}")
    
    # Load alignment summary
    summary_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__alignment_summary.yaml"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            outputs['alignment_summary'] = yaml.safe_load(f)
        if verbose:
            print("✅ Loaded alignment summary")
    else:
        print(f"⚠️  Alignment summary not found: {summary_path}")
        outputs['alignment_summary'] = {}
    
    # Load original cell positions (needed to recreate regions)
    pheno_pos_path = merge_fp / "cell_positions" / f"P-{plate}_W-{well}__phenotype_cell_positions.parquet"
    if pheno_pos_path.exists():
        outputs['phenotype_positions'] = pd.read_parquet(pheno_pos_path)
        if verbose:
            print(f"✅ Loaded phenotype positions: {len(outputs['phenotype_positions'])} cells")
    else:
        raise FileNotFoundError(f"Phenotype positions not found: {pheno_pos_path}")
    
    sbs_pos_path = merge_fp / "cell_positions" / f"P-{plate}_W-{well}__sbs_cell_positions.parquet"
    if sbs_pos_path.exists():
        outputs['sbs_positions'] = pd.read_parquet(sbs_pos_path)
        if verbose:
            print(f"✅ Loaded SBS positions: {len(outputs['sbs_positions'])} cells")
    else:
        raise FileNotFoundError(f"SBS positions not found: {sbs_pos_path}")
    
    # Load scaled phenotype positions
    scaled_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_scaled.parquet"
    if scaled_path.exists():
        outputs['phenotype_scaled'] = pd.read_parquet(scaled_path)
        if verbose:
            print(f"✅ Loaded scaled phenotype positions: {len(outputs['phenotype_scaled'])} cells")
    else:
        raise FileNotFoundError(f"Scaled phenotype positions not found: {scaled_path}")
    
    # Load transformed phenotype positions  
    transformed_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    if transformed_path.exists():
        outputs['phenotype_transformed'] = pd.read_parquet(transformed_path)
        if verbose:
            print(f"✅ Loaded transformed phenotype positions: {len(outputs['phenotype_transformed'])} cells")
    else:
        raise FileNotFoundError(f"Transformed phenotype positions not found: {transformed_path}")
    
    return outputs

def display_well_alignment_summary(alignment_data):
    """Display a summary of the well alignment results.
    
    Args:
        alignment_data (dict): Output from load_well_alignment_outputs
    """
    alignment_params = alignment_data['alignment_params'].iloc[0]
    summary = alignment_data.get('alignment_summary', {})
    
    print("=" * 60)
    print("WELL ALIGNMENT SUMMARY")
    print("=" * 60)
    
    # Basic info
    print(f"Plate: {summary.get('plate', 'Unknown')}")
    print(f"Well: {summary.get('well', 'Unknown')}")
    
    # Scale factor and overlap
    print(f"\nCoordinate Scaling:")
    print(f"  Scale factor: {summary.get('scale_factor', 'Unknown'):.6f}")
    print(f"  Overlap fraction: {summary.get('overlap_fraction', 0):.1%}")
    
    # Triangle hashing
    if verbose:
        print(f"\nTriangle Hashing:")
        print(f"  Phenotype triangles: {summary.get('phenotype_triangles', 0):,}")
        print(f"  SBS triangles: {summary.get('sbs_triangles', 0):,}")
    
    # Alignment results
    alignment_info = summary.get('alignment', {})
    print(f"\nAlignment Results:")
    print(f"  Approach: {alignment_info.get('approach', 'Unknown')}")
    print(f"  Transformation: {alignment_info.get('transformation_type', 'Unknown')}")
    print(f"  Score: {alignment_info.get('score', 0):.3f}")
    print(f"  Determinant: {alignment_info.get('determinant', 1):.6f}")
    print(f"  Region size: {alignment_info.get('region_size', 'Unknown')}")
    print(f"  Attempts: {alignment_info.get('attempts', 'Unknown')}")
    
    # Cell counts
    print(f"\nCell Counts:")
    print(f"  Original phenotype: {len(alignment_data.get('phenotype_positions', []))}")
    print(f"  Original SBS: {len(alignment_data.get('sbs_positions', []))}")
    print(f"  Scaled phenotype: {len(alignment_data.get('phenotype_scaled', []))}")
    print(f"  Transformed phenotype: {len(alignment_data.get('phenotype_transformed', []))}")
    
    print("=" * 60)