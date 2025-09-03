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


def plot_sbs_ph_matching_heatmap(
    df_merge,
    df_info,
    target="sbs",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plots the rate of matching segmented cells between phenotype and SBS datasets by well and tile in a convenient plate layout.

    Args:
        df_merge: DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of merge_triangle_hash. Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to SBS data.
        df_info: DataFrame of all cells segmented from either phenotype or SBS images, e.g., concatenated outputs for all tiles
            and wells of extract_phenotype_minimal(data_phenotype=nulcei, nuclei=nuclei), often used as `sbs_cell_info`
            rule in Snakemake.
        target: Which dataset to use as the target, e.g., if target='sbs', plots the fraction of cells in each SBS tile
            that match to a phenotype cell. Should match the information stored in df_info; if df_info is a table of all
            segmented cells from SBS tiles, then target should be set as 'sbs'.
        shape: Shape of subplot for each well used in `plot_plate_heatmap`. Defaults to 'square' and infers shape based on
            the value of `target`.
        plate: Plate type for `plot_plate_heatmap`, options are {'6W', '24W', '96W'}.
        return_plot: If True, returns `df_summary`.
        return_summary: If True, returns `df_summary`.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        df_summary: DataFrame used for plotting, returned if `return_summary=True`.
        axes: Numpy array of matplotlib Axes objects.
    """
    # Determine the merge columns and source based on the target
    if target == "sbs":
        merge_cols = ["site", "cell_1"]
        source = "phenotype"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_sbs"
    elif target == "phenotype":
        merge_cols = ["tile", "cell_0"]
        source = "sbs"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_ph"
    else:
        raise ValueError("target = {} not implemented".format(target))

    # Calculate the summary dataframe
    df_summary = (
        df_info.rename(columns={"tile": merge_cols[0], "cell": merge_cols[1]})[
            ["well"] + merge_cols
        ]
        .merge(
            df_merge[["well"] + merge_cols + ["distance"]],
            how="left",
            on=["well"] + merge_cols,
        )
        .assign(matched=lambda x: x["distance"].notna())
        .groupby(["well"] + merge_cols[:1])["matched"]
        .value_counts(normalize=True)
        .rename("fraction of {} cells matched to {} cells".format(target, source))
        .to_frame()
        .reset_index()
        .query("matched==True")
        .drop(columns="matched")
        .rename(columns={merge_cols[0]: "tile"})
    )

    if return_summary and return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes[0]
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return axes[0]
    elif return_summary:
        return df_summary
    else:
        return None


def plot_channel_histogram(df_before, df_after, channel_min_cutoff=0):
    """Generates a histogram of channel values with raw counts and consistent bin edges.

    Args:
        df_before: DataFrame containing channel values before cleaning.
        df_after: DataFrame containing channel values after cleaning.
        channel_min_cutoff: Threshold value to mark with a red vertical line. Defaults to 0.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))

    # Calculate bin edges based on the full range of data
    min_val = min(df_before["channels_min"].min(), df_after["channels_min"].min())
    max_val = max(df_before["channels_min"].max(), df_after["channels_min"].max())
    bins = np.linspace(min_val, max_val, 201)  # 201 edges make 200 bins

    # Plot histograms with raw counts instead of density
    plt.hist(
        df_before["channels_min"].dropna(),
        bins=bins,
        color="blue",
        alpha=0.5,
        label="Before clean",
    )
    plt.hist(
        df_after["channels_min"].dropna(),
        bins=bins,
        color="orange",
        alpha=0.5,
        label="After clean",
    )

    # Add vertical line for channel_min_cutoff
    plt.axvline(channel_min_cutoff, color="red", linestyle="--", label="Cutoff")

    plt.title("Histogram of channels_min Values")
    plt.xlabel("channels_min")
    plt.ylabel("Count")
    plt.legend()
    return fig


def plot_cell_positions(df_merge, title, color=None, hue="channels_min"):
    """Generates a scatter plot of cell positions in the i_0, j_0 coordinate space.

    Args:
        df_merge: DataFrame containing cell position data with i_0, j_0 columns.
        title: Plot title.
        color: Fixed color for all points. If specified, overrides hue.
        hue: Column name for color variation. Defaults to 'channels_min'.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 20))

    # Plot scatter with either fixed color or hue-based coloring
    if color is not None:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", color=color, alpha=0.5)
    else:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", hue=hue, alpha=0.5)

    plt.title(title)
    plt.xlabel("i_0")
    plt.ylabel("j_0")
    return fig


def load_well_alignment_outputs(root_fp, plate, well):
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
        print(f"✅ Loaded alignment parameters: {len(outputs['alignment_params'])} entries")
    else:
        raise FileNotFoundError(f"Alignment parameters not found: {alignment_path}")
    
    # Load alignment summary
    summary_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__alignment_summary.yaml"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            outputs['alignment_summary'] = yaml.safe_load(f)
        print("✅ Loaded alignment summary")
    else:
        print(f"⚠️  Alignment summary not found: {summary_path}")
        outputs['alignment_summary'] = {}
    
    # Load original cell positions (needed to recreate regions)
    pheno_pos_path = merge_fp / "cell_positions" / f"P-{plate}_W-{well}__phenotype_cell_positions.parquet"
    if pheno_pos_path.exists():
        outputs['phenotype_positions'] = pd.read_parquet(pheno_pos_path)
        print(f"✅ Loaded phenotype positions: {len(outputs['phenotype_positions'])} cells")
    else:
        raise FileNotFoundError(f"Phenotype positions not found: {pheno_pos_path}")
    
    sbs_pos_path = merge_fp / "cell_positions" / f"P-{plate}_W-{well}__sbs_cell_positions.parquet"
    if sbs_pos_path.exists():
        outputs['sbs_positions'] = pd.read_parquet(sbs_pos_path)
        print(f"✅ Loaded SBS positions: {len(outputs['sbs_positions'])} cells")
    else:
        raise FileNotFoundError(f"SBS positions not found: {sbs_pos_path}")
    
    # Load scaled phenotype positions
    scaled_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_scaled.parquet"
    if scaled_path.exists():
        outputs['phenotype_scaled'] = pd.read_parquet(scaled_path)
        print(f"✅ Loaded scaled phenotype positions: {len(outputs['phenotype_scaled'])} cells")
    else:
        raise FileNotFoundError(f"Scaled phenotype positions not found: {scaled_path}")
    
    # Load transformed phenotype positions  
    transformed_path = merge_fp / "well_alignment" / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    if transformed_path.exists():
        outputs['phenotype_transformed'] = pd.read_parquet(transformed_path)
        print(f"✅ Loaded transformed phenotype positions: {len(outputs['phenotype_transformed'])} cells")
    else:
        raise FileNotFoundError(f"Transformed phenotype positions not found: {transformed_path}")
    
    return outputs


def recreate_alignment_regions(alignment_data):
    """Recreate the same regional sampling used during well alignment.
    
    This function re-runs the adaptive sampling loop exactly as performed during 
    triangle_hash_well_alignment, ensuring the same region is used for visualization.
    
    Args:
        alignment_data (dict): Output from load_well_alignment_outputs
        
    Returns:
        tuple: (phenotype_region, sbs_region, final_region_size)
    """
    from lib.merge.well_alignment import sample_region_for_alignment, well_level_triangle_hash
    
    # Get the scaled phenotype positions (same as used during alignment)
    phenotype_scaled = alignment_data['phenotype_scaled']
    sbs_positions = alignment_data['sbs_positions']
    
    print(f"Recreating adaptive alignment regions...")
    print(f"Available cells: {len(phenotype_scaled)} phenotype, {len(sbs_positions)} SBS")
    
    # Parameters from triangle_hash_well_alignment
    initial_region_size = 7000
    min_triangles = 100
    max_attempts = 5
    expansion_factor = 1.5
    
    current_region_size = initial_region_size
    pheno_region = pd.DataFrame()  # Initialize empty
    sbs_region = pd.DataFrame()    # Initialize empty
    
    # Adaptive region sampling loop (same as in triangle_hash_well_alignment)
    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}: trying region size {current_region_size:.0f}px")
        
        # Sample regions using current size
        try:
            pheno_region, sbs_region, region_info = sample_region_for_alignment(
                phenotype_resized=phenotype_scaled,
                sbs_positions=sbs_positions,
                region_size=int(current_region_size),
                strategy="center"
            )
            
            if pheno_region.empty or sbs_region.empty:
                print(f"    Empty regions, expanding...")
                current_region_size *= expansion_factor
                continue
            
            print(f"    Regions: {len(pheno_region)} phenotype, {len(sbs_region)} SBS cells")
            
            # Check if we have enough cells for triangle generation
            if len(pheno_region) < 10 or len(sbs_region) < 10:
                print(f"    Insufficient cells, expanding...")
                current_region_size *= expansion_factor
                continue
            
            # Test triangle generation (same check as in alignment)
            try:
                pheno_triangles = well_level_triangle_hash(pheno_region)
                sbs_triangles = well_level_triangle_hash(sbs_region)
                
                if len(pheno_triangles) < min_triangles or len(sbs_triangles) < min_triangles:
                    print(f"    Insufficient triangles ({len(pheno_triangles)}, {len(sbs_triangles)}), expanding...")
                    current_region_size *= expansion_factor
                    continue
                
                print(f"    ✅ Success: {len(pheno_triangles)} phenotype triangles, {len(sbs_triangles)} SBS triangles")
                return pheno_region, sbs_region, current_region_size
                
            except Exception as e:
                print(f"    Triangle generation failed: {e}, expanding...")
                current_region_size *= expansion_factor
                continue
                
        except Exception as e:
            print(f"    Region sampling failed: {e}, expanding...")
            current_region_size *= expansion_factor
            continue
    
    # Fallback: if all attempts failed, return full datasets but warn user
    print(f"  ⚠️  All region attempts failed, using full datasets as fallback")
    print(f"     This may not represent the exact regions used during alignment")
    
    # Return the full scaled datasets as a last resort
    return (
        alignment_data['phenotype_scaled'], 
        alignment_data['sbs_positions'], 
        current_region_size
    )


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
    print(f"Status: {summary.get('status', 'Unknown')}")
    
    # Scale factor and overlap
    print(f"\nCoordinate Scaling:")
    print(f"  Scale factor: {summary.get('scale_factor', 'Unknown'):.6f}")
    print(f"  Overlap fraction: {summary.get('overlap_fraction', 0):.1%}")
    
    # Triangle hashing
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
    
    # Plot merge example using regional sampling
    print("\n2. Plotting merge example with regional sampling...")
    plot_well_merge_example(alignment_data, threshold=threshold, sample_size=1000)
    
    # NEW: Display merged cells data instead of simulated regional sampling
    print("\n3. Displaying merged cell matches...")
    display_merged_cells_for_site(root_fp, plate, well, selected_site, distance_threshold, max_display_rows)
    
    return alignment_data


def display_merged_cells_for_site(root_fp, plate, well, selected_site=None, 
                                 distance_threshold=15.0, max_display_rows=1000):
    """Display merged cell data from the actual parquet file.
    
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
    
    # Construct path to merged cells file
    root_path = Path(root_fp)
    merge_fp = root_path / "merge"
    merged_cells_path = merge_fp / "well_cell_merge" / f"P-{plate}_W-{well}__raw_matches.parquet"
    
    if not merged_cells_path.exists():
        print(f"❌ Merged cells file not found: {merged_cells_path}")
        print("   Make sure well cell merging has been completed for this plate/well")
        return
    
    try:
        # Load merged cells data
        print(f"📁 Loading merged cells data from: {merged_cells_path}")
        merged_df = pd.read_parquet(merged_cells_path)
        print(f"✅ Loaded {len(merged_df)} total cell matches")
        
        # Get available sites
        available_sites = sorted(merged_df['site'].unique()) if 'site' in merged_df.columns else []
        print(f"📍 Available sites: {available_sites}")
        
        # Select site to display
        if selected_site is None:
            if available_sites:
                selected_site = available_sites[0]
                print(f"🎯 Auto-selected site: {selected_site}")
            else:
                print("❌ No sites found in merged data")
                return
        elif selected_site not in available_sites:
            print(f"❌ Selected site '{selected_site}' not found. Available: {available_sites}")
            return
        else:
            print(f"🎯 Using selected site: {selected_site}")
        
        # Filter data by site and distance threshold
        site_data = merged_df[merged_df['site'] == selected_site].copy()
        filtered_data = site_data[site_data['distance'] <= distance_threshold].copy()
        
        print(f"🔍 Site '{selected_site}' statistics:")
        print(f"   Total matches: {len(site_data)}")
        print(f"   Within {distance_threshold}px: {len(filtered_data)}")
        print(f"   Match rate within threshold: {len(filtered_data)/len(site_data)*100:.1f}%")
        
        if len(filtered_data) == 0:
            print(f"⚠️  No matches found within {distance_threshold}px threshold")
            return
        
        # Calculate statistics
        distances = filtered_data['distance']
        print(f"   Distance statistics:")
        print(f"     Mean: {distances.mean():.2f}px")
        print(f"     Median: {distances.median():.2f}px")
        print(f"     Min: {distances.min():.2f}px")
        print(f"     Max: {distances.max():.2f}px")
        print(f"     Within 5px: {(distances <= 5).sum()} ({(distances <= 5).sum()/len(distances)*100:.1f}%)")
        print(f"     Within 10px: {(distances <= 10).sum()} ({(distances <= 10).sum()/len(distances)*100:.1f}%)")
        
        # Limit display rows for performance
        if len(filtered_data) > max_display_rows:
            display_data = filtered_data.head(max_display_rows).copy()
            print(f"📊 Displaying first {max_display_rows} matches (out of {len(filtered_data)})")
        else:
            display_data = filtered_data.copy()
            print(f"📊 Displaying all {len(display_data)} matches")
        
        # Format the data for display
        display_columns = [
            'plate', 'well', 'site', 'tile', 'cell_0', 'cell_1', 
            'i_0', 'j_0', 'i_1', 'j_1', 'area_0', 'area_1', 'distance',
            'stitched_cell_id_0', 'stitched_cell_id_1'
        ]
        
        # Only show columns that exist in the data
        existing_columns = [col for col in display_columns if col in display_data.columns]
        display_df = display_data[existing_columns].copy()
        
        # Round numerical columns for better display
        numerical_cols = ['i_0', 'j_0', 'i_1', 'j_1', 'area_0', 'area_1', 'distance']
        for col in numerical_cols:
            if col in display_df.columns:
                if col == 'distance':
                    display_df[col] = display_df[col].round(2)
                else:
                    display_df[col] = display_df[col].round(1)
        
        # Display the table
        print("\n" + "="*120)
        print(f"MERGED CELL MATCHES - SITE: {selected_site}")
        print(f"Distance Threshold: ≤{distance_threshold}px | Showing: {len(display_df)} matches")
        print("="*120)
        
        # Set pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        print(display_df.to_string(index=False))
        
        # Reset pandas options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        print("="*120)
        
        # Create visualization of match quality
        create_match_quality_visualization(filtered_data, selected_site, distance_threshold)
        
    except Exception as e:
        print(f"❌ Error loading or processing merged cells data: {e}")
        import traceback
        traceback.print_exc()


def create_match_quality_visualization(merged_data, site, distance_threshold):
    """Create visualization of match quality for the merged cells.
    
    Args:
        merged_data (pd.DataFrame): Filtered merged cell data
        site (str): Site name for title
        distance_threshold (float): Distance threshold used
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if len(merged_data) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Match Quality Analysis - Site: {site}', fontsize=16)
    
    # 1. Distance distribution histogram
    ax1 = axes[0, 0]
    distances = merged_data['distance']
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
    
    # 2. Scatter plot of cell positions (SBS coordinates)
    ax2 = axes[0, 1]
    if len(merged_data) <= 5000:  # Only plot if reasonable number of points
        scatter = ax2.scatter(merged_data['j_0'], merged_data['i_0'], 
                            c=merged_data['distance'], s=20, alpha=0.6, 
                            cmap='viridis', edgecolors='white', linewidths=0.3)
        plt.colorbar(scatter, ax=ax2, label='Distance (px)')
        ax2.set_xlabel('j₀ (SBS coordinates)')
        ax2.set_ylabel('i₀ (SBS coordinates)')
        ax2.set_title('Cell Positions (colored by match distance)')
        ax2.invert_yaxis()  # Match image coordinates
    else:
        ax2.text(0.5, 0.5, f'Too many points to plot\n({len(merged_data)} matches)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Cell Positions (too many to display)')
    
    # 3. Area comparison
    ax3 = axes[1, 0]
    if 'area_0' in merged_data.columns and 'area_1' in merged_data.columns:
        ax3.scatter(merged_data['area_0'], merged_data['area_1'], 
                   c=merged_data['distance'], s=15, alpha=0.6, cmap='viridis')
        # Add diagonal line for reference
        min_area = min(merged_data['area_0'].min(), merged_data['area_1'].min())
        max_area = max(merged_data['area_0'].max(), merged_data['area_1'].max())
        ax3.plot([min_area, max_area], [min_area, max_area], 'r--', alpha=0.5, label='Equal areas')
        ax3.set_xlabel('Area₀ (SBS)')
        ax3.set_ylabel('Area₁ (Phenotype)')
        ax3.set_title('Cell Area Comparison')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Area data not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Cell Area Comparison')
    
    # 4. Match quality summary
    ax4 = axes[1, 1]
    
    # Distance bins for quality assessment
    quality_bins = [
        ('Excellent\n(≤2px)', (distances <= 2).sum()),
        ('Very Good\n(2-5px)', ((distances > 2) & (distances <= 5)).sum()),
        ('Good\n(5-10px)', ((distances > 5) & (distances <= 10)).sum()),
        ('Fair\n(10-15px)', ((distances > 10) & (distances <= 15)).sum()),
        ('Poor\n(>15px)', (distances > 15).sum())
    ]
    
    labels, counts = zip(*quality_bins)
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    
    # Filter out zero counts for cleaner display
    non_zero = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if non_zero:
        labels_nz, counts_nz, colors_nz = zip(*non_zero)
        
        wedges, texts, autotexts = ax4.pie(counts_nz, labels=labels_nz, colors=colors_nz,
                                          autopct='%1.1f%%', startangle=90)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    ax4.set_title('Match Quality Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n📈 MATCH QUALITY SUMMARY")
    print(f"   Total matches analyzed: {len(merged_data)}")
    for label, count in quality_bins:
        if count > 0:
            percentage = count / len(merged_data) * 100
            print(f"   {label.replace(chr(10), ' ')}: {count} ({percentage:.1f}%)")


def load_merged_cells_interactive(root_fp, plate, well):
    """Interactive function to explore merged cells data with different parameters.
    
    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
    """
    from pathlib import Path
    import pandas as pd
    
    # Load the data first
    merged_cells_path = Path(root_fp) / "merge" / "well_cell_merge" / f"P-{plate}_W-{well}__raw_matches.parquet"
    
    if not merged_cells_path.exists():
        print(f"❌ Merged cells file not found: {merged_cells_path}")
        return None
    
    try:
        merged_df = pd.read_parquet(merged_cells_path)
        available_sites = sorted(merged_df['site'].unique()) if 'site' in merged_df.columns else []
        
        print(f"🔍 Merged Cells Explorer for Plate {plate}, Well {well}")
        print(f"📊 Total matches: {len(merged_df)}")
        print(f"📍 Available sites: {available_sites}")
        
        def explore_site(site=None, max_distance=15.0, max_rows=500):
            """Explore a specific site with given parameters."""
            return display_merged_cells_for_site(root_fp, plate, well, site, max_distance, max_rows)
        
        return merged_df, explore_site
        
    except Exception as e:
        print(f"❌ Error loading merged cells data: {e}")
        return None


def plot_well_merge_example(alignment_data, threshold, sample_size=None, figsize=(30, 10)):
    """Custom version of well merge visualization that avoids the sklearn issue.
    
    This bypasses the problematic plot_merge_example function and creates the 
    visualization directly using matplotlib.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    
    # Recreate the same regions used during alignment
    df_ph_region, df_sbs_region, final_region_size = recreate_alignment_regions(alignment_data)
    
    if df_ph_region.empty or df_sbs_region.empty:
        print("❌ Failed to recreate alignment regions")
        return
    
    print(f"Using final region size: {final_region_size:.0f}px")
    print(f"Region cells: {len(df_ph_region)} phenotype, {len(df_sbs_region)} SBS")
    
    # Sample data if requested
    if sample_size:
        if len(df_ph_region) > sample_size:
            df_ph_region = df_ph_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} phenotype cells for visualization")
        
        if len(df_sbs_region) > sample_size:
            df_sbs_region = df_sbs_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} SBS cells for visualization")
    
    # Get transformation parameters
    alignment_params = alignment_data['alignment_params'].iloc[0]
    rotation_flat = alignment_params['rotation_matrix_flat']
    translation_list = alignment_params['translation_vector']
    
    rotation = np.array(rotation_flat).reshape(2, 2)
    translation = np.array(translation_list)
    
    print(f"Applying transformation:")
    print(f"  Rotation: {rotation}")
    print(f"  Translation: {translation}")
    print(f"  Score: {alignment_params['score']:.3f}")
    print(f"  Determinant: {alignment_params['determinant']:.6f}")
    
    # Extract coordinates
    ph_coords = df_ph_region[['i', 'j']].values
    sbs_coords = df_sbs_region[['i', 'j']].values
    
    # Apply transformation to phenotype coordinates
    ph_transformed = ph_coords @ rotation.T + translation
    
    # Calculate distances and matches
    distances = cdist(ph_transformed, sbs_coords, metric='euclidean')
    min_distances = distances.min(axis=1)
    closest_indices = distances.argmin(axis=1)
    
    # Find matches within threshold
    matches = min_distances <= threshold
    match_count = matches.sum()
    match_rate = match_count / len(ph_coords) if len(ph_coords) > 0 else 0
    
    print(f"Match results:")
    print(f"  Matches found: {match_count}/{len(ph_coords)} ({match_rate:.1%})")
    print(f"  Mean distance: {min_distances.mean():.2f}")
    print(f"  Median distance: {np.median(min_distances):.2f}")
    
    # Create the visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Original phenotype positions (blue)
    ax1 = axes[0]
    ax1.scatter(ph_coords[:, 0], ph_coords[:, 1], c='blue', alpha=0.6, s=20, label='Phenotype')
    ax1.set_title(f'Original Phenotype Positions\n({len(ph_coords)} cells)')
    ax1.set_xlabel('i (pixels)')
    ax1.set_ylabel('j (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Transformed phenotype (red) + SBS (green)
    ax2 = axes[1] 
    ax2.scatter(sbs_coords[:, 0], sbs_coords[:, 1], c='green', alpha=0.6, s=20, label='SBS')
    ax2.scatter(ph_transformed[:, 0], ph_transformed[:, 1], c='red', alpha=0.6, s=20, label='Phenotype (transformed)')
    ax2.set_title(f'Transformed Phenotype + SBS Positions\nScore: {alignment_params["score"]:.3f}, Det: {alignment_params["determinant"]:.6f}')
    ax2.set_xlabel('i (pixels)')
    ax2.set_ylabel('j (pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Matches visualization
    ax3 = axes[2]
    
    # Plot unmatched cells in light colors
    unmatched = ~matches
    if unmatched.sum() > 0:
        ax3.scatter(ph_transformed[unmatched, 0], ph_transformed[unmatched, 1], 
                   c='lightcoral', alpha=0.4, s=15, label=f'Unmatched phenotype ({unmatched.sum()})')
    
    # Plot SBS cells
    ax3.scatter(sbs_coords[:, 0], sbs_coords[:, 1], c='lightgreen', alpha=0.4, s=15, label=f'SBS ({len(sbs_coords)})')
    
    # Plot matched pairs with connecting lines
    if match_count > 0:
        matched_ph = ph_transformed[matches]
        matched_sbs_indices = closest_indices[matches]
        matched_sbs = sbs_coords[matched_sbs_indices]
        
        # Draw connecting lines
        for i in range(len(matched_ph)):
            ax3.plot([matched_ph[i, 0], matched_sbs[i, 0]], 
                    [matched_ph[i, 1], matched_sbs[i, 1]], 
                    'gray', alpha=0.3, linewidth=0.5)
        
        # Plot matched points
        ax3.scatter(matched_ph[:, 0], matched_ph[:, 1], c='red', alpha=0.8, s=25, 
                   label=f'Matched phenotype ({match_count})')
        ax3.scatter(matched_sbs[:, 0], matched_sbs[:, 1], c='green', alpha=0.8, s=25,
                   label=f'Matched SBS ({match_count})')
    
    ax3.set_title(f'Cell Matches (threshold = {threshold})\nMatch rate: {match_rate:.1%} ({match_count}/{len(ph_coords)})')
    ax3.set_xlabel('i (pixels)')
    ax3.set_ylabel('j (pixels)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, (match_count, len(ph_coords), match_rate)

    
class StitchQC:
    """Quality control class for stitched image analysis."""
    
    def __init__(self, base_path, plate, well):
        """Initialize QC for a specific plate/well.

        Parameters:
        -----------
        base_path : str or Path
            Base path to your analysis outputs (should point to the 'merge' directory)
        plate : str/int
            Plate identifier
        well : str
            Well identifier (e.g., 'A3')
        """
        self.base_path = Path(base_path)
        self.plate = str(plate)
        self.well = well
        prefix = f"P-{plate}_W-{well}__"

        # Define expected file paths based on your actual structure
        self.phenotype_image = (
            self.base_path / "stitched_images" / f"{prefix}phenotype_stitched_image.npy"
        )
        self.phenotype_mask = (
            self.base_path / "stitched_masks" / f"{prefix}phenotype_stitched_mask.npy"
        )
        self.phenotype_positions = (
            self.base_path
            / "cell_positions"
            / f"{prefix}phenotype_cell_positions.parquet"
        )
        self.phenotype_overlay = (
            self.base_path / "overlays" / f"{prefix}phenotype_overlay.png"
        )

        self.sbs_image = (
            self.base_path / "stitched_images" / f"{prefix}sbs_stitched_image.npy"
        )
        self.sbs_mask = (
            self.base_path / "stitched_masks" / f"{prefix}sbs_stitched_mask.npy"
        )
        self.sbs_positions = (
            self.base_path / "cell_positions" / f"{prefix}sbs_cell_positions.parquet"
        )
        self.sbs_overlay = self.base_path / "overlays" / f"{prefix}sbs_overlay.png"

        print(f"Initialized QC for Plate {plate}, Well {well}")
        print(f"Base path: {self.base_path}")
        print(f"Looking for files with prefix: {prefix}")
        print(f"Example phenotype image path: {self.phenotype_image}")
        self.check_files()

    def check_files(self):
        """Check which output files exist."""
        files = {
            "Phenotype Image": self.phenotype_image,
            "Phenotype Mask": self.phenotype_mask,
            "Phenotype Positions": self.phenotype_positions,
            "Phenotype Overlay": self.phenotype_overlay,
            "SBS Image": self.sbs_image,
            "SBS Mask": self.sbs_mask,
            "SBS Positions": self.sbs_positions,
            "SBS Overlay": self.sbs_overlay,
        }

        print("\n=== File Status ===")
        for name, path in files.items():
            status = "✅ EXISTS" if path.exists() else "❌ MISSING"
            size = f"({path.stat().st_size / 1e6:.1f} MB)" if path.exists() else ""
            print(f"{name:20} {status} {size}")

    def view_overlays(self, figsize=(15, 6)):
        """Display phenotype and SBS overlays side by side."""
        from skimage import io
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Phenotype overlay
        if self.phenotype_overlay.exists():
            ph_overlay = io.imread(self.phenotype_overlay)
            axes[0].imshow(ph_overlay)
            axes[0].set_title(
                f"Phenotype Overlay\nPlate {self.plate}, Well {self.well}"
            )
            axes[0].axis("off")
        else:
            axes[0].text(
                0.5,
                0.5,
                "Phenotype\nOverlay\nMissing",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
                fontsize=16,
            )
            axes[0].set_title("Phenotype Overlay - MISSING")

        # SBS overlay
        if self.sbs_overlay.exists():
            sbs_overlay = io.imread(self.sbs_overlay)
            axes[1].imshow(sbs_overlay)
            axes[1].set_title(f"SBS Overlay\nPlate {self.plate}, Well {self.well}")
            axes[1].axis("off")
        else:
            axes[1].text(
                0.5,
                0.5,
                "SBS\nOverlay\nMissing",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=16,
            )
            axes[1].set_title("SBS Overlay - MISSING")

        plt.tight_layout()
        plt.show()

    def analyze_cell_positions(self):
        """Analyze cell position data and create summary plots."""
        # Load position data
        ph_pos = None
        sbs_pos = None

        if self.phenotype_positions.exists():
            ph_pos = pd.read_parquet(self.phenotype_positions)
            print(f"Phenotype: {len(ph_pos)} cells")

        if self.sbs_positions.exists():
            sbs_pos = pd.read_parquet(self.sbs_positions)
            print(f"SBS: {len(sbs_pos)} cells")

        if ph_pos is None and sbs_pos is None:
            print("No position data available")
            return

        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Cell Position Analysis - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )

        # Plot 1: Cell counts by modality
        counts = []
        labels = []
        if ph_pos is not None:
            counts.append(len(ph_pos))
            labels.append("Phenotype")
        if sbs_pos is not None:
            counts.append(len(sbs_pos))
            labels.append("SBS")

        axes[0, 0].bar(labels, counts, color=["skyblue", "lightcoral"])
        axes[0, 0].set_title("Cell Counts by Modality")
        axes[0, 0].set_ylabel("Number of Cells")
        for i, count in enumerate(counts):
            axes[0, 0].text(i, count + max(counts) * 0.01, str(count), ha="center")

        # Plot 2: Cell area distributions
        if ph_pos is not None:
            axes[0, 1].hist(
                ph_pos["area"], bins=50, alpha=0.7, label="Phenotype", color="skyblue"
            )
        if sbs_pos is not None:
            axes[0, 1].hist(
                sbs_pos["area"], bins=50, alpha=0.7, label="SBS", color="lightcoral"
            )
        axes[0, 1].set_title("Cell Area Distributions")
        axes[0, 1].set_xlabel("Cell Area (pixels)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # Plot 3: Tile distribution (if available)
        if ph_pos is not None and "tile" in ph_pos.columns:
            tile_counts = ph_pos["tile"].value_counts().sort_index()
            axes[0, 2].bar(
                tile_counts.index, tile_counts.values, color="skyblue", alpha=0.7
            )
            axes[0, 2].set_title("Phenotype Cells per Tile")
            axes[0, 2].set_xlabel("Tile ID")
            axes[0, 2].set_ylabel("Cell Count")
        else:
            axes[0, 2].text(
                0.5,
                0.5,
                "No Tile\nData",
                ha="center",
                va="center",
                transform=axes[0, 2].transAxes,
                fontsize=14,
            )
            axes[0, 2].set_title("Phenotype Tiles - No Data")

        # Plot 4: Spatial distribution - Phenotype
        if ph_pos is not None:
            scatter = axes[1, 0].scatter(
                ph_pos["j"],
                ph_pos["i"],
                c=ph_pos.get("tile", 0),
                s=1,
                alpha=0.6,
                cmap="tab10",
            )
            axes[1, 0].set_title("Phenotype Cell Positions")
            axes[1, 0].set_xlabel("J (Column)")
            axes[1, 0].set_ylabel("I (Row)")
            axes[1, 0].invert_yaxis()  # Match image coordinates
            if "tile" in ph_pos.columns:
                plt.colorbar(scatter, ax=axes[1, 0], label="Tile ID")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No Phenotype\nPosition Data",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=14,
            )

        # Plot 5: Spatial distribution - SBS
        if sbs_pos is not None:
            scatter = axes[1, 1].scatter(
                sbs_pos["j"],
                sbs_pos["i"],
                c=sbs_pos.get("tile", 0),
                s=1,
                alpha=0.6,
                cmap="tab10",
            )
            axes[1, 1].set_title("SBS Cell Positions")
            axes[1, 1].set_xlabel("J (Column)")
            axes[1, 1].set_ylabel("I (Row)")
            axes[1, 1].invert_yaxis()  # Match image coordinates
            if "tile" in sbs_pos.columns:
                plt.colorbar(scatter, ax=axes[1, 1], label="Tile ID")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No SBS\nPosition Data",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
            )

        # Plot 6: Position overlay comparison
        if ph_pos is not None and sbs_pos is not None:
            axes[1, 2].scatter(
                ph_pos["j"],
                ph_pos["i"],
                s=1,
                alpha=0.5,
                label="Phenotype",
                color="blue",
            )
            axes[1, 2].scatter(
                sbs_pos["j"], sbs_pos["i"], s=1, alpha=0.5, label="SBS", color="red"
            )
            axes[1, 2].set_title("Position Overlay Comparison")
            axes[1, 2].set_xlabel("J (Column)")
            axes[1, 2].set_ylabel("I (Row)")
            axes[1, 2].invert_yaxis()
            axes[1, 2].legend()
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Missing Data\nfor Comparison",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
                fontsize=14,
            )

        plt.tight_layout()
        plt.show()

        return ph_pos, sbs_pos

    def check_stitching_quality_efficient(self, sample_region=None, preview_downsample=20, 
                                        brightness_range=(0.1, 2.0), contrast_range=(0.5, 3.0)):
        """Memory-efficient stitching quality check with interactive brightness/contrast controls.
        
        Parameters:
        -----------
        sample_region : tuple, optional
            (start_i, end_i, start_j, end_j) region to examine at full resolution
        preview_downsample : int, default 20
            Downsampling factor for full well preview (higher = lower memory usage)
        brightness_range : tuple, default (0.1, 2.0)
            Min and max values for brightness adjustment
        contrast_range : tuple, default (0.5, 3.0)
            Min and max values for contrast adjustment
        """
        # Check matplotlib backend
        backend = plt.get_backend()
        print(f"Matplotlib backend: {backend}")
        if 'inline' in backend.lower():
            print("Warning: Interactive widgets may not work with 'inline' backend.")
            print("Try running: %matplotlib widget")
            print("Or: %matplotlib qt")
        
        # Create figure with space for sliders
        fig = plt.figure(figsize=(16, 14))
        
        # Create main subplot area (leave space at bottom for sliders)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.4, bottom=0.1)
        axes = [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        ]
        
        # Slider area - make them bigger and more spaced out
        slider_ax1 = plt.axes([0.15, 0.05, 0.25, 0.03])  # [left, bottom, width, height]
        slider_ax2 = plt.axes([0.55, 0.05, 0.25, 0.03])
        
        fig.suptitle(
            f"Stitching Quality Check - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )

        # Store image data and display objects for slider updates
        image_data = {}
        display_objects = {}
        
        def adjust_image_display(img_array, brightness=1.0, contrast=1.0):
            """Apply brightness and contrast adjustments."""
            # Normalize to 0-1 range first
            img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            # Apply contrast (multiply) then brightness (add)
            adjusted = np.clip(contrast * img_norm + (brightness - 1.0), 0, 1)
            return adjusted
        
        def update_display(val=None):
            """Update all image displays with current slider values."""
            brightness = brightness_slider.val
            contrast = contrast_slider.val
            
            for key, img_data in image_data.items():
                adjusted = adjust_image_display(img_data, brightness, contrast)
                display_objects[key].set_data(adjusted)
            
            fig.canvas.draw_idle()

        # Process phenotype image
        if self.phenotype_image.exists():
            # Memory map the array instead of loading it
            ph_img = np.load(self.phenotype_image, mmap_mode='r')
            print(f"Phenotype image shape: {ph_img.shape}")
            print(f"Estimated size: {ph_img.nbytes / 1e9:.1f} GB")

            # Create downsampled preview without loading full image
            ph_preview = ph_img[::preview_downsample, ::preview_downsample]
            image_data['ph_preview'] = ph_preview
            
            # Initial display with default brightness/contrast
            ph_preview_adj = adjust_image_display(ph_preview)
            display_objects['ph_preview'] = axes[0][0].imshow(ph_preview_adj, cmap="gray")
            axes[0][0].set_title(f"Phenotype Full Well\n(downsampled {preview_downsample}x)")
            axes[0][0].axis("off")

            # Handle sample region
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                
                # Validate region bounds
                start_i = max(0, min(start_i, ph_img.shape[0]))
                end_i = max(start_i, min(end_i, ph_img.shape[0]))
                start_j = max(0, min(start_j, ph_img.shape[1]))
                end_j = max(start_j, min(end_j, ph_img.shape[1]))
                
                # Extract only the requested region (memory efficient)
                ph_sample = np.array(ph_img[start_i:end_i, start_j:end_j])
                image_data['ph_sample'] = ph_sample
                
                ph_sample_adj = adjust_image_display(ph_sample)
                display_objects['ph_sample'] = axes[0][1].imshow(ph_sample_adj, cmap="gray")
                axes[0][1].set_title(
                    f"Phenotype Sample Region\n[{start_i}:{end_i}, {start_j}:{end_j}]\n"
                    f"Size: {ph_sample.shape}"
                )

                # Add rectangle to preview showing sample region
                rect = Rectangle(
                    (start_j // preview_downsample, start_i // preview_downsample),
                    (end_j - start_j) // preview_downsample,
                    (end_i - start_i) // preview_downsample,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                axes[0][0].add_patch(rect)
            else:
                # Show center region
                h, w = ph_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(1000, min(h, w) // 8)  # Larger default region
                
                start_i = center_h - size
                end_i = center_h + size
                start_j = center_w - size
                end_j = center_w + size
                
                ph_center = np.array(ph_img[start_i:end_i, start_j:end_j])
                image_data['ph_sample'] = ph_center
                
                ph_center_adj = adjust_image_display(ph_center)
                display_objects['ph_sample'] = axes[0][1].imshow(ph_center_adj, cmap="gray")
                axes[0][1].set_title(f"Phenotype Center Region\n{ph_center.shape}")
            
            axes[0][1].axis("off")

        # Process SBS image
        if self.sbs_image.exists():
            # Memory map the SBS array
            sbs_img = np.load(self.sbs_image, mmap_mode='r')
            print(f"SBS image shape: {sbs_img.shape}")
            print(f"Estimated size: {sbs_img.nbytes / 1e9:.1f} GB")

            # Create downsampled preview
            sbs_downsample = max(1, max(sbs_img.shape) // 1000)
            sbs_preview = sbs_img[::sbs_downsample, ::sbs_downsample]
            image_data['sbs_preview'] = sbs_preview

            sbs_preview_adj = adjust_image_display(sbs_preview)
            display_objects['sbs_preview'] = axes[1][0].imshow(sbs_preview_adj, cmap="gray")
            axes[1][0].set_title(f"SBS Full Well\n(downsampled {sbs_downsample}x)")
            axes[1][0].axis("off")

            # Handle sample region for SBS
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                
                # Calculate scale factor between phenotype and SBS
                if self.phenotype_image.exists():
                    scale_h = sbs_img.shape[0] / ph_img.shape[0]
                    scale_w = sbs_img.shape[1] / ph_img.shape[1]
                else:
                    scale_h = scale_w = 0.25  # Default assumption for 10x vs 40x
                
                # Scale region coordinates for SBS
                sbs_start_i = max(0, int(start_i * scale_h))
                sbs_end_i = min(sbs_img.shape[0], int(end_i * scale_h))
                sbs_start_j = max(0, int(start_j * scale_w))
                sbs_end_j = min(sbs_img.shape[1], int(end_j * scale_w))

                sbs_sample = np.array(sbs_img[sbs_start_i:sbs_end_i, sbs_start_j:sbs_end_j])
                image_data['sbs_sample'] = sbs_sample
                
                sbs_sample_adj = adjust_image_display(sbs_sample)
                display_objects['sbs_sample'] = axes[1][1].imshow(sbs_sample_adj, cmap="gray")
                axes[1][1].set_title(
                    f"SBS Sample Region\n[{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}]\n"
                    f"Size: {sbs_sample.shape}"
                )
            else:
                # Show center region
                h, w = sbs_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(500, min(h, w) // 4)
                
                sbs_center = np.array(sbs_img[center_h - size:center_h + size, 
                                            center_w - size:center_w + size])
                image_data['sbs_sample'] = sbs_center
                
                sbs_center_adj = adjust_image_display(sbs_center)
                display_objects['sbs_sample'] = axes[1][1].imshow(sbs_center_adj, cmap="gray")
                axes[1][1].set_title(f"SBS Center Region\n{sbs_center.shape}")
            
            axes[1][1].axis("off")

        # Create brightness and contrast sliders with larger, more visible controls
        brightness_slider = Slider(
            slider_ax1, 'Brightness', 
            brightness_range[0], brightness_range[1], 
            valinit=1.0, valstep=0.05, valfmt='%.2f',
            facecolor='lightblue', edgecolor='black'
        )
        contrast_slider = Slider(
            slider_ax2, 'Contrast', 
            contrast_range[0], contrast_range[1], 
            valinit=1.0, valstep=0.05, valfmt='%.2f',
            facecolor='lightgreen', edgecolor='black'
        )
        
        # Connect sliders to update function
        brightness_slider.on_changed(update_display)
        contrast_slider.on_changed(update_display)

        # Add text instructions
        fig.text(0.5, 0.01, 'Drag sliders to adjust brightness and contrast', 
                ha='center', fontsize=12, style='italic')

        plt.tight_layout()
        plt.show()
        
        # Keep references to prevent garbage collection
        fig._brightness_slider = brightness_slider
        fig._contrast_slider = contrast_slider
        
        return fig, brightness_slider, contrast_slider

    def check_stitching_quality_static(self, sample_region=None, preview_downsample=20,
                                     brightness_levels=[0.3, 0.7, 1.0, 1.5, 2.0]):
        """Non-interactive version showing multiple brightness levels side by side.
        Use this if interactive sliders don't work.
        
        Parameters:
        -----------
        sample_region : tuple, optional
            (start_i, end_i, start_j, end_j) region to examine at full resolution
        preview_downsample : int, default 20
            Downsampling factor for full well preview
        brightness_levels : list, default [0.3, 0.7, 1.0, 1.5, 2.0]
            Different brightness levels to display
        """
        # Handle single brightness level case
        if len(brightness_levels) == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 10))
            # Convert to 2D indexing for consistency
            axes = axes.reshape(2, 1)
        else:
            fig, axes = plt.subplots(2, len(brightness_levels), figsize=(4*len(brightness_levels), 8))
            # Ensure axes is always 2D
            if len(brightness_levels) == 1:
                axes = axes.reshape(2, 1)
        
        fig.suptitle(
            f"Brightness Comparison - Plate {self.plate}, Well {self.well}",
            fontsize=16,
        )
        
        def adjust_image_display(img_array, brightness=1.0):
            """Apply brightness adjustment."""
            img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            adjusted = np.clip(img_norm * brightness, 0, 1)
            return adjusted

        # Process phenotype image
        if self.phenotype_image.exists():
            ph_img = np.load(self.phenotype_image, mmap_mode='r')
            
            if sample_region:
                start_i, end_i, start_j, end_j = sample_region
                start_i = max(0, min(start_i, ph_img.shape[0]))
                end_i = max(start_i, min(end_i, ph_img.shape[0]))
                start_j = max(0, min(start_j, ph_img.shape[1]))
                end_j = max(start_j, min(end_j, ph_img.shape[1]))
                ph_sample = np.array(ph_img[start_i:end_i, start_j:end_j])
                print(f"Phenotype region: [{start_i}:{end_i}, {start_j}:{end_j}], shape: {ph_sample.shape}")
            else:
                h, w = ph_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(1000, min(h, w) // 8)
                ph_sample = np.array(ph_img[center_h - size:center_h + size, 
                                        center_w - size:center_w + size])
            
            # Show different brightness levels
            for i, brightness in enumerate(brightness_levels):
                adjusted = adjust_image_display(ph_sample, brightness)
                axes[0, i].imshow(adjusted, cmap="gray")
                axes[0, i].set_title(f"Phenotype\nBrightness: {brightness}")
                axes[0, i].axis("off")

        # Process SBS image if available
        if self.sbs_image.exists():
            sbs_img = np.load(self.sbs_image, mmap_mode='r')
            
            if sample_region:
                # Scale coordinates for SBS
                if self.phenotype_image.exists():
                    scale_h = sbs_img.shape[0] / ph_img.shape[0]
                    scale_w = sbs_img.shape[1] / ph_img.shape[1]
                else:
                    scale_h = scale_w = 0.25
                
                sbs_start_i = max(0, int(start_i * scale_h))
                sbs_end_i = min(sbs_img.shape[0], int(end_i * scale_h))
                sbs_start_j = max(0, int(start_j * scale_w))
                sbs_end_j = min(sbs_img.shape[1], int(end_j * scale_w))
                sbs_sample = np.array(sbs_img[sbs_start_i:sbs_end_i, sbs_start_j:sbs_end_j])
                print(f"SBS region: [{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}], shape: {sbs_sample.shape}")
            else:
                h, w = sbs_img.shape
                center_h, center_w = h // 2, w // 2
                size = min(500, min(h, w) // 4)
                sbs_sample = np.array(sbs_img[center_h - size:center_h + size, 
                                            center_w - size:center_w + size])
            
            # Show different brightness levels
            for i, brightness in enumerate(brightness_levels):
                adjusted = adjust_image_display(sbs_sample, brightness)
                axes[1, i].imshow(adjusted, cmap="gray")
                axes[1, i].set_title(f"SBS\nBrightness: {brightness}")
                axes[1, i].axis("off")
        
        plt.tight_layout()
        plt.show()

    def view_region(self, center_row, center_col, size=1000, brightness=2.0):
        """View a square region centered at specified coordinates with fixed brightness.
        
        Parameters:
        -----------
        center_row, center_col : int
            Center coordinates of region to view
        size : int, default 1000
            Size of square region (will be size x size pixels)
        brightness : float, default 2.0
            Brightness multiplier for display
        """
        half_size = size // 2
        start_i = center_row - half_size
        end_i = center_row + half_size
        start_j = center_col - half_size
        end_j = center_col + half_size
        
        print(f"Viewing {size}x{size} region centered at ({center_row}, {center_col}) with brightness {brightness}")
        
        # Use static version with single brightness level
        self.check_stitching_quality_static(
            sample_region=(start_i, end_i, start_j, end_j),
            brightness_levels=[brightness]
        )

    def view_mask_region(self, center_row, center_col, size=1000, modality="phenotype", 
                        adaptive_colormap=True, colormap="nipy_spectral", exclude_background=True):
        """View a square region from the stitched mask for the given modality.
        Now with adaptive colormap scaling for better visualization.

        Parameters
        ----------
        center_row, center_col : int
            Center coordinates of region to view (in pixel space of stitched image/mask)
        size : int, default 1000
            Size of square region (size x size pixels)
        modality : str, default "phenotype"
            Which mask to view: "phenotype" or "sbs"
        adaptive_colormap : bool, default True
            If True, adjusts colormap range to the actual cell ID range in the region
        colormap : str, default "nipy_spectral"
            Matplotlib colormap to use. Options: "nipy_spectral", "tab20", "Set1", "viridis", etc.
        exclude_background : bool, default True
            If True, excludes background pixels (ID=0) from colormap scaling
        """
        # Calculate initial region bounds from center coordinates
        half_size = size // 2
        start_i = center_row - half_size
        end_i = center_row + half_size
        start_j = center_col - half_size
        end_j = center_col + half_size

        if modality == "phenotype":
            mask_path = self.phenotype_mask
            title = f"Phenotype Mask Region - Plate {self.plate}, Well {self.well}"
            # For phenotype, coordinates should match directly (no scaling needed)
            final_start_i, final_end_i = start_i, end_i
            final_start_j, final_end_j = start_j, end_j
            
        elif modality == "sbs":
            mask_path = self.sbs_mask
            title = f"SBS Mask Region - Plate {self.plate}, Well {self.well}"
            
            # Load mask to get its actual dimensions for scaling
            mask = np.load(mask_path, mmap_mode="r")
            
            # Apply scaling logic - but use mask dimensions, not image dimensions
            if self.phenotype_mask.exists():
                # Load phenotype mask to calculate scale factors
                ph_mask = np.load(self.phenotype_mask, mmap_mode='r')
                
                scale_h = mask.shape[0] / ph_mask.shape[0]
                scale_w = mask.shape[1] / ph_mask.shape[1]
                
                print(f"Scaling SBS coordinates using mask dimensions:")
                print(f"  Phenotype mask shape: {ph_mask.shape}")
                print(f"  SBS mask shape: {mask.shape}")
                print(f"  Scale factors: h={scale_h:.4f}, w={scale_w:.4f}")
                
            else:
                # Fallback: try using image dimensions if masks don't exist
                if self.phenotype_image.exists():
                    ph_img = np.load(self.phenotype_image, mmap_mode='r')
                    scale_h = mask.shape[0] / ph_img.shape[0]
                    scale_w = mask.shape[1] / ph_img.shape[1]
                    print(f"Scaling using image dimensions as fallback:")
                    print(f"  Phenotype image shape: {ph_img.shape}")
                    print(f"  SBS mask shape: {mask.shape}")
                    print(f"  Scale factors: h={scale_h:.4f}, w={scale_w:.4f}")
                else:
                    scale_h = scale_w = 0.25  # Default assumption
                    print(f"Using default scale factors: h={scale_h}, w={scale_w}")
            
            # Scale region coordinates for SBS
            final_start_i = int(start_i * scale_h)
            final_end_i = int(end_i * scale_h)
            final_start_j = int(start_j * scale_w)
            final_end_j = int(end_j * scale_w)
            
            print(f"  Original region: [{start_i}:{end_i}, {start_j}:{end_j}]")
            print(f"  Scaled region: [{final_start_i}:{final_end_i}, {final_start_j}:{final_end_j}]")
            
        else:
            raise ValueError("modality must be 'phenotype' or 'sbs'")

        if not mask_path.exists():
            print(f"❌ Mask file not found: {mask_path}")
            return

        print(f"Viewing {size}x{size} mask region centered at ({center_row}, {center_col})")

        # Load mask using memory mapping (consistent with image loading approach)
        # Note: For SBS, mask was already loaded above for scaling calculations
        if modality == "phenotype":
            mask = np.load(mask_path, mmap_mode="r")
        print(f"Mask shape: {mask.shape}")

        # Validate and clip region bounds to mask dimensions
        final_start_i = max(0, min(final_start_i, mask.shape[0]))
        final_end_i = max(final_start_i, min(final_end_i, mask.shape[0]))
        final_start_j = max(0, min(final_start_j, mask.shape[1]))
        final_end_j = max(final_start_j, min(final_end_j, mask.shape[1]))

        # Extract region (now using numpy array conversion for consistency)
        region = np.array(mask[final_start_i:final_end_i, final_start_j:final_end_j])
        
        print(f"Final extracted region shape: {region.shape}")
        print(f"Final bounds used: [{final_start_i}:{final_end_i}, {final_start_j}:{final_end_j}]")
        
        # Analyze the cell ID values in this region
        unique_values = np.unique(region)
        print(f"Unique mask values in region: {len(unique_values)} values")
        print(f"Cell ID range: {unique_values.min()} to {unique_values.max()}")
        
        # Count cells vs background
        background_pixels = (region == 0).sum()
        cell_pixels = (region > 0).sum()
        print(f"Background pixels: {background_pixels:,} ({background_pixels/region.size:.1%})")
        print(f"Cell pixels: {cell_pixels:,} ({cell_pixels/region.size:.1%})")

        # Create visualization with adaptive colormap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine colormap limits
        if adaptive_colormap:
            if exclude_background and len(unique_values) > 1:
                # Use range of cell IDs only (exclude background=0)
                cell_values = unique_values[unique_values > 0]
                if len(cell_values) > 0:
                    vmin, vmax = cell_values.min(), cell_values.max()
                    print(f"Adaptive colormap range: {vmin} to {vmax} (excluding background)")
                else:
                    vmin, vmax = unique_values.min(), unique_values.max()
                    print(f"No cells found, using full range: {vmin} to {vmax}")
            else:
                vmin, vmax = unique_values.min(), unique_values.max()
                print(f"Adaptive colormap range: {vmin} to {vmax} (including background)")
        else:
            vmin, vmax = None, None
            print("Using default colormap scaling")
        
        # Display the mask with adaptive colormap
        im = ax.imshow(region, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax)
        
        ax.set_title(f"{title}\nCenter: ({center_row}, {center_col}), Region: {region.shape}\n"
                    f"Cell ID range in region: {unique_values.min()} - {unique_values.max()}")
        ax.axis("off")
        
        # Add colorbar with better labeling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if adaptive_colormap and exclude_background:
            cbar.set_label('Cell ID (background=0 excluded from scale)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Cell ID', rotation=270, labelpad=20)
        
        # Add some text info on the plot
        info_text = f"Cells: {len(unique_values)-1 if 0 in unique_values else len(unique_values)}\n"
        info_text += f"Background: {background_pixels/region.size:.1%}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print some additional useful info
        if len(unique_values) > 1:
            print(f"\nRegional statistics:")
            print(f"  Number of unique cells: {len(unique_values)-1 if 0 in unique_values else len(unique_values)}")
            print(f"  Cell density: {cell_pixels/region.size:.1%} of region")
            print(f"  Average cell ID: {unique_values[unique_values>0].mean():.0f}")
        
        # Return the region for further analysis if needed
        return region

    def compare_mask_regions(self, center_row, center_col, size=7000, colormap="nipy_spectral", 
                            adaptive_colormap=True, exclude_background=True):
        """Compare phenotype and SBS mask regions side by side with consistent colormap settings.
        
        Parameters
        ----------
        center_row, center_col : int
            Center coordinates (in phenotype space)
        size : int, default 7000
            Region size
        colormap : str, default "nipy_spectral"
            Colormap to use for both
        adaptive_colormap : bool, default True
            Use adaptive colormap scaling
        exclude_background : bool, default True
            Exclude background from colormap scaling
        """
        print(f"COMPARING MASK REGIONS AT ({center_row}, {center_col}) ± {size//2}")
        print("="*60)
        
        print("\n1. Phenotype Mask:")
        ph_region = self.view_mask_region(center_row, center_col, size, "phenotype", 
                                        adaptive_colormap, colormap, exclude_background)
        
        print("\n2. SBS Mask:")
        sbs_region = self.view_mask_region(center_row, center_col, size, "sbs",
                                        adaptive_colormap, colormap, exclude_background)
        
        # Summary comparison
        if ph_region is not None and sbs_region is not None:
            ph_cells = len(np.unique(ph_region)) - (1 if 0 in np.unique(ph_region) else 0)
            sbs_cells = len(np.unique(sbs_region)) - (1 if 0 in np.unique(sbs_region) else 0)
            
            print(f"\n" + "="*60)
            print("REGION COMPARISON SUMMARY")
            print("="*60)
            print(f"Phenotype cells in region: {ph_cells}")
            print(f"SBS cells in region: {sbs_cells}")
            print(f"Cell count ratio (SBS/PH): {sbs_cells/ph_cells:.2f}" if ph_cells > 0 else "N/A")
            print("="*60)
        
        return ph_region, sbs_region
    
    
    def view_alignment_region(self, center_row, center_col, size=7000, threshold=15.0, sample_size=1000):
        """View alignment quality for a specific region with the same coordinate system as view_region/view_mask_region.
        Shows original positions, transformed positions, and matches in the specified region.

        Parameters
        ----------
        center_row, center_col : int
            Center coordinates of region to view (in phenotype coordinate space)
        size : int, default 7000
            Size of square region (size x size pixels)
        threshold : float, default 15.0
            Distance threshold for considering cells matched
        sample_size : int, default 1000
            Maximum number of cells to plot for performance (None for all cells)
        """
        print(f"Loading alignment data for region centered at ({center_row}, {center_col})...")
        
        # Load alignment outputs
        try:
            alignment_data = load_well_alignment_outputs(self.base_path.parent, self.plate, self.well)
        except FileNotFoundError as e:
            print(f"❌ Could not load alignment data: {e}")
            print("Make sure well alignment has been run for this plate/well")
            return
        
        # Get alignment parameters
        alignment_params = alignment_data['alignment_params'].iloc[0]
        rotation_flat = alignment_params['rotation_matrix_flat']
        translation_list = alignment_params['translation_vector']
        
        rotation = np.array(rotation_flat).reshape(2, 2)
        translation = np.array(translation_list)
        
        print(f"Alignment parameters:")
        print(f"  Score: {alignment_params['score']:.3f}")
        print(f"  Determinant: {alignment_params['determinant']:.6f}")
        print(f"  Translation: {translation}")
        
        # Define region bounds (same logic as view_mask_region)
        half_size = size // 2
        start_i = center_row - half_size
        end_i = center_row + half_size
        start_j = center_col - half_size
        end_j = center_col + half_size
        
        print(f"Analyzing region: [{start_i}:{end_i}, {start_j}:{end_j}]")
        
        # Filter phenotype cells to region
        phenotype_scaled = alignment_data['phenotype_scaled']
        ph_region = phenotype_scaled[
            (phenotype_scaled['i'] >= start_i) & (phenotype_scaled['i'] <= end_i) &
            (phenotype_scaled['j'] >= start_j) & (phenotype_scaled['j'] <= end_j)
        ].copy()
        
        print(f"Phenotype cells in region: {len(ph_region)}")
        
        if len(ph_region) == 0:
            print("❌ No phenotype cells found in this region")
            return
        
        # Calculate SBS region bounds using the same scaling as view_mask_region
        if self.phenotype_image.exists() and self.sbs_image.exists():
            ph_img = np.load(self.phenotype_image, mmap_mode='r')
            sbs_img = np.load(self.sbs_image, mmap_mode='r')
            scale_h = sbs_img.shape[0] / ph_img.shape[0]
            scale_w = sbs_img.shape[1] / ph_img.shape[1]
        else:
            scale_h = scale_w = 0.25  # Default
        
        # Scale region bounds for SBS
        sbs_start_i = int(start_i * scale_h)
        sbs_end_i = int(end_i * scale_h)
        sbs_start_j = int(start_j * scale_w)
        sbs_end_j = int(end_j * scale_w)
        
        print(f"SBS region bounds (scaled): [{sbs_start_i}:{sbs_end_i}, {sbs_start_j}:{sbs_end_j}]")
        
        # Filter SBS cells to scaled region
        sbs_positions = alignment_data['sbs_positions']
        sbs_region = sbs_positions[
            (sbs_positions['i'] >= sbs_start_i) & (sbs_positions['i'] <= sbs_end_i) &
            (sbs_positions['j'] >= sbs_start_j) & (sbs_positions['j'] <= sbs_end_j)
        ].copy()
        
        print(f"SBS cells in scaled region: {len(sbs_region)}")
        
        if len(sbs_region) == 0:
            print("❌ No SBS cells found in scaled region")
            return
        
        # Sample data if requested for performance
        if sample_size and len(ph_region) > sample_size:
            ph_region = ph_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} phenotype cells for visualization")
        
        if sample_size and len(sbs_region) > sample_size:
            sbs_region = sbs_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} SBS cells for visualization")
        
        # Extract coordinates
        ph_coords = ph_region[['i', 'j']].values
        sbs_coords = sbs_region[['i', 'j']].values
        
        # Apply transformation to phenotype coordinates
        ph_transformed = ph_coords @ rotation.T + translation
        
        # Calculate distances and matches
        distances = cdist(ph_transformed, sbs_coords, metric='euclidean')
        min_distances = distances.min(axis=1)
        closest_indices = distances.argmin(axis=1)
        
        # Find matches within threshold
        matches = min_distances <= threshold
        match_count = matches.sum()
        match_rate = match_count / len(ph_coords) if len(ph_coords) > 0 else 0
        
        print(f"Match results in this region:")
        print(f"  Matches found: {match_count}/{len(ph_coords)} ({match_rate:.1%})")
        print(f"  Mean distance: {min_distances.mean():.2f} pixels")
        print(f"  Median distance: {np.median(min_distances):.2f} pixels")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Alignment Quality Check - Plate {self.plate}, Well {self.well}\n'
                    f'Region: ({center_row}, {center_col}) ± {size//2}', fontsize=14)
        
        # Panel 1: Original phenotype positions (blue) - in phenotype coordinate space
        axes[0, 0].scatter(ph_coords[:, 1], ph_coords[:, 0], c='blue', alpha=0.6, s=20, label='Phenotype')
        axes[0, 0].set_title(f'Original Phenotype Positions\n({len(ph_coords)} cells)')
        axes[0, 0].set_xlabel('j (pixels)')
        axes[0, 0].set_ylabel('i (pixels)')
        axes[0, 0].invert_yaxis()
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: SBS positions (green) - in SBS coordinate space
        axes[0, 1].scatter(sbs_coords[:, 1], sbs_coords[:, 0], c='green', alpha=0.6, s=20, label='SBS')
        axes[0, 1].set_title(f'SBS Positions\n({len(sbs_coords)} cells)')
        axes[0, 1].set_xlabel('j (pixels)')
        axes[0, 1].set_ylabel('i (pixels)')
        axes[0, 1].invert_yaxis()
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Transformed phenotype (red) + SBS (green) - both in SBS coordinate space
        axes[1, 0].scatter(sbs_coords[:, 1], sbs_coords[:, 0], c='green', alpha=0.6, s=20, label='SBS')
        axes[1, 0].scatter(ph_transformed[:, 1], ph_transformed[:, 0], c='red', alpha=0.6, s=20, label='Phenotype (transformed)')
        axes[1, 0].set_title(f'Overlay: Transformed Phenotype + SBS\nScore: {alignment_params["score"]:.3f}, Det: {alignment_params["determinant"]:.6f}')
        axes[1, 0].set_xlabel('j (pixels)')
        axes[1, 0].set_ylabel('i (pixels)')
        axes[1, 0].invert_yaxis()
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 4: Matches visualization with connecting lines
        ax = axes[1, 1]
        
        # Plot unmatched cells in light colors
        unmatched = ~matches
        if unmatched.sum() > 0:
            ax.scatter(ph_transformed[unmatched, 1], ph_transformed[unmatched, 0], 
                    c='lightcoral', alpha=0.4, s=15, label=f'Unmatched phenotype ({unmatched.sum()})')
        
        # Plot all SBS cells
        ax.scatter(sbs_coords[:, 1], sbs_coords[:, 0], c='lightgreen', alpha=0.4, s=15, label=f'SBS ({len(sbs_coords)})')
        
        # Plot matched pairs with connecting lines
        if match_count > 0:
            matched_ph = ph_transformed[matches]
            matched_sbs_indices = closest_indices[matches]
            matched_sbs = sbs_coords[matched_sbs_indices]
            
            # Draw connecting lines
            for i in range(len(matched_ph)):
                ax.plot([matched_ph[i, 1], matched_sbs[i, 1]], 
                    [matched_ph[i, 0], matched_sbs[i, 0]], 
                    'gray', alpha=0.3, linewidth=0.5)
            
            # Plot matched points on top
            ax.scatter(matched_ph[:, 1], matched_ph[:, 0], c='red', alpha=0.8, s=25, 
                    label=f'Matched phenotype ({match_count})')
            ax.scatter(matched_sbs[:, 1], matched_sbs[:, 0], c='green', alpha=0.8, s=25,
                    label=f'Matched SBS ({match_count})')
        
        ax.set_title(f'Cell Matches (threshold = {threshold}px)\nMatch rate: {match_rate:.1%} ({match_count}/{len(ph_coords)})')
        ax.set_xlabel('j (pixels)')
        ax.set_ylabel('i (pixels)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Return summary statistics
        alignment_stats = {
            'region_center': (center_row, center_col),
            'region_size': size,
            'phenotype_cells': len(ph_coords),
            'sbs_cells': len(sbs_coords),
            'matches': match_count,
            'match_rate': match_rate,
            'mean_distance': min_distances.mean(),
            'median_distance': np.median(min_distances),
            'alignment_score': alignment_params['score'],
            'determinant': alignment_params['determinant']
        }
        
        return fig, alignment_stats

    def view_alignment_and_masks(self, center_row, center_col, size=7000, threshold=15.0, 
                                brightness=2.0, sample_size=1000):
        """Comprehensive alignment check showing images, masks, and alignment quality in one view.
        
        Parameters
        ----------
        center_row, center_col : int
            Center coordinates of region to view (in phenotype coordinate space)
        size : int, default 7000
            Size of square region (size x size pixels)
        threshold : float, default 15.0
            Distance threshold for considering cells matched
        brightness : float, default 2.0
            Brightness multiplier for image display
        sample_size : int, default 1000
            Maximum number of cells to plot for performance
        """
        print(f"Comprehensive alignment check for region ({center_row}, {center_col}) ± {size//2}")
        print("-" * 80)
        
        # First show the alignment quality
        print("1. Alignment Quality Analysis:")
        fig1, stats = self.view_alignment_region(center_row, center_col, size, threshold, sample_size)
        
        print(f"\n2. Image regions (brightness = {brightness}):")
        # Show the corresponding image regions
        self.view_region(center_row, center_col, size, brightness)
        
        print(f"\n3. Mask regions:")
        # Show the corresponding mask regions
        print("Phenotype mask:")
        ph_mask_region = self.view_mask_region(center_row, center_col, size, "phenotype")
        print("SBS mask:")
        sbs_mask_region = self.view_mask_region(center_row, center_col, size, "sbs")
        
        # Print summary
        print("\n" + "="*60)
        print("REGION ALIGNMENT SUMMARY")
        print("="*60)
        print(f"Region: ({center_row}, {center_col}) ± {size//2} pixels")
        print(f"Cells found: {stats['phenotype_cells']} phenotype, {stats['sbs_cells']} SBS")
        print(f"Match rate: {stats['match_rate']:.1%} ({stats['matches']}/{stats['phenotype_cells']})")
        print(f"Mean distance: {stats['mean_distance']:.2f} pixels")
        print(f"Alignment score: {stats['alignment_score']:.3f}")
        print(f"Determinant: {stats['determinant']:.6f}")
        
        # Assess quality
        if stats['match_rate'] > 0.8 and stats['mean_distance'] < threshold/2:
            print("✅ EXCELLENT alignment in this region")
        elif stats['match_rate'] > 0.6 and stats['mean_distance'] < threshold:
            print("🟡 GOOD alignment in this region") 
        elif stats['match_rate'] > 0.3:
            print("🟠 FAIR alignment in this region")
        else:
            print("❌ POOR alignment in this region")
        
        print("="*60)
        
        return stats, ph_mask_region, sbs_mask_region

    def check_alignment_at_multiple_regions(self, regions, threshold=15.0, sample_size=500):
        """Check alignment quality at multiple regions to get a comprehensive view.
        
        Parameters
        ----------
        regions : list of tuples
            List of (center_row, center_col, size) tuples defining regions to check
        threshold : float, default 15.0
            Distance threshold for matches
        sample_size : int, default 500
            Maximum cells per region for performance
            
        Returns
        -------
        DataFrame : Summary of alignment quality across all regions
        """
        print("MULTI-REGION ALIGNMENT ANALYSIS")
        print("="*80)
        
        results = []
        
        for i, (center_row, center_col, size) in enumerate(regions):
            print(f"\nRegion {i+1}: ({center_row}, {center_col}) ± {size//2}")
            print("-" * 40)
            
            try:
                stats, _, _ = self.view_alignment_and_masks(
                    center_row, center_col, size, threshold, 
                    brightness=1.5, sample_size=sample_size
                )
                results.append(stats)
                
            except Exception as e:
                print(f"❌ Failed to analyze region {i+1}: {e}")
                # Add failed entry
                results.append({
                    'region_center': (center_row, center_col),
                    'region_size': size,
                    'phenotype_cells': 0,
                    'sbs_cells': 0, 
                    'matches': 0,
                    'match_rate': 0,
                    'mean_distance': np.nan,
                    'median_distance': np.nan,
                    'alignment_score': np.nan,
                    'determinant': np.nan
                })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL REGIONS")
        print("="*80)
        print(summary_df[['region_center', 'phenotype_cells', 'sbs_cells', 'match_rate', 'mean_distance']].to_string(index=False))
        
        # Overall assessment
        overall_match_rate = summary_df['match_rate'].mean()
        overall_mean_distance = summary_df['mean_distance'].mean()
        
        print(f"\nOverall statistics:")
        print(f"  Average match rate: {overall_match_rate:.1%}")
        print(f"  Average mean distance: {overall_mean_distance:.2f} pixels")
        print(f"  Regions with >80% match rate: {(summary_df['match_rate'] > 0.8).sum()}/{len(summary_df)}")
        
        if overall_match_rate > 0.8 and overall_mean_distance < threshold/2:
            print("✅ EXCELLENT overall alignment quality")
        elif overall_match_rate > 0.6 and overall_mean_distance < threshold:
            print("🟡 GOOD overall alignment quality") 
        elif overall_match_rate > 0.3:
            print("🟠 FAIR overall alignment quality - consider re-alignment")
        else:
            print("❌ POOR overall alignment quality - re-alignment recommended")
        
        print("="*80)
        
        return summary_df

    def get_mask_info(self, modality="phenotype"):
        """Get basic information about a mask including dimensions and suggested viewing coordinates.
        
        Parameters
        ----------
        modality : str, default "phenotype"
            Which mask to examine: "phenotype" or "sbs"
            
        Returns
        -------
        dict : Information about the mask
        """
        if modality == "phenotype":
            mask_path = self.phenotype_mask
        elif modality == "sbs":
            mask_path = self.sbs_mask
        else:
            raise ValueError("modality must be 'phenotype' or 'sbs'")
        
        if not mask_path.exists():
            print(f"❌ Mask file not found: {mask_path}")
            return None
        
        # Load mask header info without loading full array
        mask = np.load(mask_path, mmap_mode='r')
        
        info = {
            'shape': mask.shape,
            'center': (mask.shape[0] // 2, mask.shape[1] // 2),
            'max_region_size': min(mask.shape) // 2,
            'path': mask_path
        }
        
        print(f"📐 Mask shape: {info['shape']}, Center: {info['center']}, Max region size: {info['max_region_size']}")
        return info


# Enable interactive backend
plt.ion()  # Turn on interactive mode


def quick_qc(base_path, plate, well):
    """Quick QC check for a single well."""
    qc = StitchQC(base_path, plate, well)
    qc.view_overlays()
    return qc.analyze_cell_positions()


def batch_qc_report(base_path, plate_wells):
    """Generate QC reports for multiple wells.

    Parameters:
    -----------
    base_path : str
        Path to analysis outputs
    plate_wells : list of tuples
        [(plate1, well1), (plate2, well2), ...]
    """
    print("BATCH QC REPORT")
    print("=" * 60)

    summary = []

    for plate, well in plate_wells:
        qc = StitchQC(base_path, plate, well)

        # Quick file check
        ph_exists = qc.phenotype_positions.exists()
        sbs_exists = qc.sbs_positions.exists()

        ph_count = 0
        sbs_count = 0

        if ph_exists:
            ph_count = len(pd.read_parquet(qc.phenotype_positions))
        if sbs_exists:
            sbs_count = len(pd.read_parquet(qc.sbs_positions))

        summary.append(
            {
                "plate": plate,
                "well": well,
                "ph_exists": ph_exists,
                "sbs_exists": sbs_exists,
                "ph_cells": ph_count,
                "sbs_cells": sbs_count,
                "status": "OK"
                if ph_exists and sbs_exists and ph_count > 0 and sbs_count > 0
                else "ISSUE",
            }
        )

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    return summary_df


def load_stitched_image(image_path: str, downsample: int = 1) -> np.ndarray:
    """Load a stitched image from .npy file with optional downsampling.
    
    Args:
        image_path: Path to the .npy stitched image file
        downsample: Downsampling factor for the image (default 4)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Load image with downsampling
    image = load_stitched_image(image_path, downsample=downsample)
    print(f"Loaded image shape: {image.shape}")
    
    # Normalize for display
    display_image = normalize_image_for_display(image)
    
    # Adjust region for downsampling
    if region is not None:
        i_min, i_max, j_min, j_max = region
        # Scale region coordinates by downsample factor
        i_min_ds, i_max_ds = i_min // downsample, i_max // downsample
        j_min_ds, j_max_ds = j_min // downsample, j_max // downsample
        display_image = display_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        print(f"Cropped to region: {region} (downsampled: [{i_min_ds}, {i_max_ds}, {j_min_ds}, {j_max_ds}])")
        # Use original coordinates for extent
        extent = [j_min, j_max, i_max, i_min]
    else:
        region = (0, image.shape[0] * downsample, 0, image.shape[1] * downsample)
        i_min, i_max, j_min, j_max = region
        extent = [j_min, j_max, i_max, i_min]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show image
    if display_image.ndim == 3:
        ax.imshow(display_image, extent=extent)
    else:
        ax.imshow(display_image, cmap='gray', extent=extent)
    
    # Overlay cell positions if provided
    if positions_path is not None:
        positions = load_cell_positions(positions_path)
        print(f"Loaded {len(positions)} cell positions")
        
        # Filter to region
        region_positions = positions[
            (positions['i'] >= i_min) & (positions['i'] <= i_max) &
            (positions['j'] >= j_min) & (positions['j'] <= j_max)
        ]
        print(f"Found {len(region_positions)} cells in region")
        
        if len(region_positions) > 0:
            if color_by_stitched_id and 'stitched_cell_id' in region_positions.columns:
                # Color by stitched_cell_id
                unique_ids = region_positions['stitched_cell_id'].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
                
                scatter = ax.scatter(region_positions['j'], region_positions['i'], 
                          c=region_positions['stitched_cell_id'], s=cell_size, 
                          alpha=0.7, edgecolors='white', linewidths=0.5,
                          cmap='tab20')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Stitched Cell ID')
                
                print(f"Colored by stitched_cell_id: {len(unique_ids)} unique IDs in region")
            else:
                # Use single color
                ax.scatter(region_positions['j'], region_positions['i'], 
                          c=cell_color, s=cell_size, alpha=0.7, edgecolors='white', linewidths=0.5)
                
                if color_by_stitched_id:
                    print("⚠️  stitched_cell_id column not found, using single color")
    
    ax.set_title(title)
    ax.set_xlabel('J (pixels)')
    ax.set_ylabel('I (pixels)')
    
    return fig


def compare_modality_images(phenotype_image_path: str,
                           sbs_image_path: str,
                           phenotype_positions_path: str,
                           sbs_positions_path: str,
                           alignment_summary_path: Optional[str] = None,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           color_by_stitched_id: bool = False,
                           figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
    """Compare stitched images from both modalities side by side.
    
    Args:
        phenotype_image_path: Path to phenotype stitched image
        sbs_image_path: Path to SBS stitched image  
        phenotype_positions_path: Path to phenotype cell positions
        sbs_positions_path: Path to SBS cell positions
        alignment_summary_path: Optional path to alignment summary
        region: Region to crop both images to
        color_by_stitched_id: If True, color cells by their stitched_cell_id
        figsize: Figure size
        
    Returns:
        Matplotlib figure with side-by-side comparison
    """
    # Load images
    pheno_image = load_stitched_image(phenotype_image_path)
    sbs_image = load_stitched_image(sbs_image_path)
    
    # Load positions
    pheno_positions = load_cell_positions(phenotype_positions_path)
    sbs_positions = load_cell_positions(sbs_positions_path)
    
    print(f"Phenotype: {pheno_image.shape} image, {len(pheno_positions)} cells")
    print(f"SBS: {sbs_image.shape} image, {len(sbs_positions)} cells")
    
    # Load alignment info if available
    alignment_info = ""
    if alignment_summary_path is not None and Path(alignment_summary_path).exists():
        summary = load_alignment_summary(alignment_summary_path)
        if 'alignment' in summary:
            align_data = summary['alignment']
            alignment_info = f"Score: {align_data.get('score', 0):.3f}, Det: {align_data.get('determinant', 1):.3f}"
    
    # Normalize images
    pheno_display = normalize_image_for_display(pheno_image)
    sbs_display = normalize_image_for_display(sbs_image)
    
    # Apply region cropping if specified
    if region is not None:
        i_min, i_max, j_min, j_max = region
        pheno_display = pheno_display[i_min:i_max, j_min:j_max]
        sbs_display = sbs_display[i_min:i_max, j_min:j_max]
        
        # Filter positions to region
        pheno_positions = pheno_positions[
            (pheno_positions['i'] >= i_min) & (pheno_positions['i'] <= i_max) &
            (pheno_positions['j'] >= j_min) & (pheno_positions['j'] <= j_max)
        ]
        sbs_positions = sbs_positions[
            (sbs_positions['i'] >= i_min) & (sbs_positions['i'] <= i_max) &
            (sbs_positions['j'] >= j_min) & (sbs_positions['j'] <= j_max)
        ]
    else:
        region = (0, min(pheno_image.shape[0], sbs_image.shape[0]), 
                 0, min(pheno_image.shape[1], sbs_image.shape[1]))
        i_min, i_max, j_min, j_max = region
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Phenotype image
    if pheno_display.ndim == 3:
        ax1.imshow(pheno_display, extent=[j_min, j_max, i_max, i_min])
    else:
        ax1.imshow(pheno_display, cmap='gray', extent=[j_min, j_max, i_max, i_min])
    
    if color_by_stitched_id and 'stitched_cell_id' in pheno_positions.columns:
        scatter1 = ax1.scatter(pheno_positions['j'], pheno_positions['i'], 
                   c=pheno_positions['stitched_cell_id'], s=3, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, cmap='tab20')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Phenotype Stitched Cell ID')
    else:
        ax1.scatter(pheno_positions['j'], pheno_positions['i'], 
                   c='red', s=3, alpha=0.8, edgecolors='white', linewidths=0.3)
    
    ax1.set_title(f'Phenotype\n{len(pheno_positions)} cells in region')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    
    # SBS image
    if sbs_display.ndim == 3:
        ax2.imshow(sbs_display, extent=[j_min, j_max, i_max, i_min])
    else:
        ax2.imshow(sbs_display, cmap='gray', extent=[j_min, j_max, i_max, i_min])
    
    if color_by_stitched_id and 'stitched_cell_id' in sbs_positions.columns:
        scatter2 = ax2.scatter(sbs_positions['j'], sbs_positions['i'], 
                   c=sbs_positions['stitched_cell_id'], s=3, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, cmap='tab20')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('SBS Stitched Cell ID')
    else:
        ax2.scatter(sbs_positions['j'], sbs_positions['i'], 
                   c='blue', s=3, alpha=0.8, edgecolors='white', linewidths=0.3)
    
    ax2.set_title(f'SBS\n{len(sbs_positions)} cells in region')
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    
    if alignment_info:
        fig.suptitle(f'Modality Comparison - {alignment_info}')
    else:
        fig.suptitle('Modality Comparison')
    
    plt.tight_layout()
    return fig


def visualize_matched_cells(phenotype_image_path: str,
                           sbs_image_path: str,
                           matched_cells_path: str,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           max_distance: float = 10.0,
                           downsample: int = 4,
                           figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
    """Visualize matched cells overlaid on both modality images.
    
    Args:
        phenotype_image_path: Path to phenotype stitched image
        sbs_image_path: Path to SBS stitched image
        matched_cells_path: Path to matched cells parquet file
        region: Region to focus on
        max_distance: Maximum distance to show matches
        downsample: Downsampling factor for images (default 4)
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing matches
    """
    # Load images with downsampling
    pheno_image = normalize_image_for_display(load_stitched_image(phenotype_image_path, downsample=downsample))
    sbs_image = normalize_image_for_display(load_stitched_image(sbs_image_path, downsample=downsample))
    
    # Load matched cells
    matches = pd.read_parquet(matched_cells_path)
    print(f"Loaded {len(matches)} matched cells")
    
    # Filter by distance if specified
    if max_distance is not None:
        matches = matches[matches['distance'] <= max_distance]
        print(f"Filtered to {len(matches)} matches within {max_distance}px")
    
    # Apply region if specified
    if region is not None:
        i_min, i_max, j_min, j_max = region
        # Scale region for downsampled images
        i_min_ds, i_max_ds = i_min // downsample, i_max // downsample
        j_min_ds, j_max_ds = j_min // downsample, j_max // downsample
        
        pheno_image = pheno_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        sbs_image = sbs_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        
        # Filter matches to region (using original coordinates)
        matches = matches[
            (matches['i_0'] >= i_min) & (matches['i_0'] <= i_max) &
            (matches['j_0'] >= j_min) & (matches['j_0'] <= j_max) &
            (matches['i_1'] >= i_min) & (matches['i_1'] <= i_max) &
            (matches['j_1'] >= j_min) & (matches['j_1'] <= j_max)
        ]
        print(f"Found {len(matches)} matches in region")
        extent = [j_min, j_max, i_max, i_min]
    else:
        region = (0, min(pheno_image.shape[0], sbs_image.shape[0]) * downsample, 
                 0, min(pheno_image.shape[1], sbs_image.shape[1]) * downsample)
        i_min, i_max, j_min, j_max = region
        extent = [j_min, j_max, i_max, i_min]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Show images
    if pheno_image.ndim == 3:
        ax1.imshow(pheno_image, extent=extent)
    else:
        ax1.imshow(pheno_image, cmap='gray', extent=extent)
    
    if sbs_image.ndim == 3:
        ax2.imshow(sbs_image, extent=extent)
    else:
        ax2.imshow(sbs_image, cmap='gray', extent=extent)
    
    # Plot matches
    if len(matches) > 0:
        # Color code by distance
        distances = matches['distance']
        scatter1 = ax1.scatter(matches['j_0'], matches['i_0'], 
                              c=distances, s=20, cmap='viridis', alpha=0.8,
                              edgecolors='white', linewidths=0.5)
        scatter2 = ax2.scatter(matches['j_1'], matches['i_1'], 
                              c=distances, s=20, cmap='viridis', alpha=0.8,
                              edgecolors='white', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label('Match Distance (pixels)')
    
    ax1.set_title(f'Phenotype Matches\n{len(matches)} cells')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    
    ax2.set_title(f'SBS Matches\n{len(matches)} cells')
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    
    if len(matches) > 0:
        mean_dist = matches['distance'].mean()
        fig.suptitle(f'Matched Cells (Mean Distance: {mean_dist:.1f}px)')
    else:
        fig.suptitle('Matched Cells (No matches in region)')
    
    plt.tight_layout()
    return fig


def plot_alignment_overview(phenotype_positions_path: str,
                           sbs_positions_path: str,
                           transformed_positions_path: Optional[str] = None,
                           alignment_summary_path: Optional[str] = None,
                           sample_size: int = 5000,
                           figsize: Tuple[int, int] = (18, 6)) -> plt.Figure:
    """Plot overview of coordinate alignment process.
    
    Args:
        phenotype_positions_path: Path to original phenotype positions
        sbs_positions_path: Path to SBS positions
        transformed_positions_path: Path to transformed phenotype positions
        alignment_summary_path: Path to alignment summary
        sample_size: Number of cells to sample for plotting
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing alignment progression
    """
    # Load data
    pheno_pos = load_cell_positions(phenotype_positions_path)
    sbs_pos = load_cell_positions(sbs_positions_path)
    
    print(f"Loaded {len(pheno_pos)} phenotype and {len(sbs_pos)} SBS positions")
    
    # Sample for plotting
    if len(pheno_pos) > sample_size:
        pheno_pos = pheno_pos.sample(n=sample_size)
    if len(sbs_pos) > sample_size:
        sbs_pos = sbs_pos.sample(n=sample_size)
    
    # Load alignment summary if available
    alignment_info = {}
    if alignment_summary_path and Path(alignment_summary_path).exists():
        summary = load_alignment_summary(alignment_summary_path)
        alignment_info = summary.get('alignment', {})
    
    # Determine number of subplots
    n_plots = 2
    if transformed_positions_path and Path(transformed_positions_path).exists():
        n_plots = 3
        transformed_pos = load_cell_positions(transformed_positions_path)
        if len(transformed_pos) > sample_size:
            transformed_pos = transformed_pos.sample(n=sample_size)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # Plot 1: Original coordinates
    ax1.scatter(pheno_pos['j'], pheno_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype')
    ax1.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
    ax1.set_title('Original Coordinates')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and show coordinate ranges
    pheno_range_i = (pheno_pos['i'].min(), pheno_pos['i'].max())
    pheno_range_j = (pheno_pos['j'].min(), pheno_pos['j'].max())
    sbs_range_i = (sbs_pos['i'].min(), sbs_pos['i'].max())
    sbs_range_j = (sbs_pos['j'].min(), sbs_pos['j'].max())
    
    ax1.text(0.02, 0.98, f'Pheno I: {pheno_range_i[0]:.0f}-{pheno_range_i[1]:.0f}\n'
                         f'Pheno J: {pheno_range_j[0]:.0f}-{pheno_range_j[1]:.0f}\n'
                         f'SBS I: {sbs_range_i[0]:.0f}-{sbs_range_i[1]:.0f}\n'
                         f'SBS J: {sbs_range_j[0]:.0f}-{sbs_range_j[1]:.0f}',
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: After scaling (use transformed if available, otherwise estimate scaling)
    if 'transformed_pos' in locals():
        scaled_pos = transformed_pos
        title2 = 'After Scaling & Transform'
    else:
        # Estimate scaling for visualization
        scale_factor = alignment_info.get('scale_factor', 1.0)
        scaled_pos = pheno_pos.copy()
        scaled_pos['i'] = scaled_pos['i'] * scale_factor
        scaled_pos['j'] = scaled_pos['j'] * scale_factor
        title2 = f'After Scaling (factor: {scale_factor:.3f})'
    
    ax2.scatter(scaled_pos['j'], scaled_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype (scaled)')
    ax2.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
    ax2.set_title(title2)
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add alignment info if available
    if alignment_info:
        info_text = f"Score: {alignment_info.get('score', 0):.3f}\n"
        info_text += f"Det: {alignment_info.get('determinant', 1):.3f}\n"
        info_text += f"Type: {alignment_info.get('transformation_type', 'unknown')}"
        
        ax2.text(0.02, 0.98, info_text,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Final transformation (if available)
    if n_plots == 3 and 'transformed_pos' in locals():
        ax3.scatter(transformed_pos['j'], transformed_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype (final)')
        ax3.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
        ax3.set_title('After Full Transformation')
        ax3.set_xlabel('J (pixels)')
        ax3.set_ylabel('I (pixels)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calculate overlap
        trans_range_i = (transformed_pos['i'].min(), transformed_pos['i'].max())
        trans_range_j = (transformed_pos['j'].min(), transformed_pos['j'].max())
        
        overlap_i = max(0, min(trans_range_i[1], sbs_range_i[1]) - max(trans_range_i[0], sbs_range_i[0]))
        overlap_j = max(0, min(trans_range_j[1], sbs_range_j[1]) - max(trans_range_j[0], sbs_range_j[0]))
        
        total_i = max(trans_range_i[1], sbs_range_i[1]) - min(trans_range_i[0], sbs_range_i[0])
        total_j = max(trans_range_j[1], sbs_range_j[1]) - min(trans_range_j[0], sbs_range_j[0])
        
        overlap_fraction = (overlap_i * overlap_j) / (total_i * total_j) if total_i > 0 and total_j > 0 else 0
        
        ax3.text(0.02, 0.98, f'Overlap: {overlap_fraction:.1%}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig


def get_merge_file_paths(plate: str, well: str, root_fp: str) -> Dict[str, Path]:
    """Get file paths using the same structure as the pipeline.
    
    Matches the approach used in run_well_alignment_qc and other pipeline functions.
    
    Args:
        plate: Plate identifier
        well: Well identifier
        root_fp: Root file path for the project
        
    Returns:
        Dictionary with all relevant file paths
    """
    from lib.shared.file_utils import get_filename
    
    root_path = Path(root_fp)
    merge_fp = root_path / "merge"
    
    # Build wildcards dict for filename generation
    wildcards = {"plate": plate, "well": well}
    
    paths = {
        # Stitched images
        "phenotype_image": merge_fp / "stitched_images" / get_filename(wildcards, "phenotype_stitched_image", "npy"),
        "sbs_image": merge_fp / "stitched_images" / get_filename(wildcards, "sbs_stitched_image", "npy"),
        
        # Stitched masks
        "phenotype_mask": merge_fp / "stitched_masks" / get_filename(wildcards, "phenotype_stitched_mask", "npy"),
        "sbs_mask": merge_fp / "stitched_masks" / get_filename(wildcards, "sbs_stitched_mask", "npy"),
        
        # Cell positions
        "phenotype_positions": merge_fp / "cell_positions" / get_filename(wildcards, "phenotype_cell_positions", "parquet"),
        "sbs_positions": merge_fp / "cell_positions" / get_filename(wildcards, "sbs_cell_positions", "parquet"),
        
        # Well alignment outputs
        "phenotype_scaled": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_scaled", "parquet"),
        "phenotype_triangles": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_triangles", "parquet"),
        "sbs_triangles": merge_fp / "well_alignment" / get_filename(wildcards, "sbs_triangles", "parquet"),
        "alignment_params": merge_fp / "well_alignment" / get_filename(wildcards, "alignment", "parquet"),
        "alignment_summary": merge_fp / "well_alignment" / get_filename(wildcards, "alignment_summary", "yaml"),
        "phenotype_transformed": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_transformed", "parquet"),
        
        # Well cell merge outputs  
        "raw_matches": merge_fp / "well_cell_merge" / get_filename(wildcards, "raw_matches", "parquet"),
        "merged_cells": merge_fp / "well_cell_merge" / get_filename(wildcards, "merged_cells", "parquet"),
        "merge_summary": merge_fp / "well_cell_merge" / get_filename(wildcards, "merge_summary", "yaml"),
        
        # Deduplication outputs
        "deduplicated_cells": merge_fp / "well_merge_deduplicate" / get_filename(wildcards, "deduplicated_cells", "parquet"),
        "dedup_summary": merge_fp / "well_merge_deduplicate" / get_filename(wildcards, "dedup_summary", "yaml"),
    }
    
    return paths


def analyze_well_merge(plate: str, well: str, root_fp: str, region: Optional[Tuple[int, int, int, int]] = None):
    """Comprehensive analysis of a well merge result using pipeline file structure.
    
    Args:
        plate: Plate identifier
        well: Well identifier  
        root_fp: Root file path for the project (same as used in pipeline)
        region: Optional region to focus analysis on (i_min, i_max, j_min, j_max)
    """
    print(f"=== ANALYZING WELL MERGE: {plate}_{well} ===")
    
    # Get file paths using pipeline structure
    paths = get_merge_file_paths(plate, well, root_fp)
    
    # Check which files exist
    existing_files = []
    for name, path in paths.items():
        if path.exists():
            existing_files.append(name)
            print(f"✅ Found: {name}")
        else:
            print(f"❌ Missing: {name}")
    
    if not existing_files:
        print("❌ No merge files found! Check your paths.")
        return
    
    # Use the specified sampling region if provided, otherwise use default
    if region is None:
        region = (9764, 16764, 9810, 16810)  # Your specified sampling region
        print(f"Using default sampling region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]")
    else:
        print(f"Using custom region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]")
    
    # 1. Show alignment overview if position files exist
    if all(f in existing_files for f in ["phenotype_positions", "sbs_positions"]):
        print("\n--- PLOTTING ALIGNMENT OVERVIEW ---")
        try:
            fig1 = plot_alignment_overview(
                str(paths["phenotype_positions"]), 
                str(paths["sbs_positions"]),
                str(paths["phenotype_transformed"]) if "phenotype_transformed" in existing_files else None,
                str(paths["alignment_summary"]) if "alignment_summary" in existing_files else None
            )
            fig1.suptitle(f'Alignment Overview - {plate}_{well}')
            plt.show()
        except Exception as e:
            print(f"❌ Error plotting alignment overview: {e}")
    
    # 2. Compare modality images if they exist
    if all(f in existing_files for f in ["phenotype_image", "sbs_image", "phenotype_positions", "sbs_positions"]):
        print("\n--- COMPARING MODALITY IMAGES ---")
        try:
            fig2 = compare_modality_images(
                str(paths["phenotype_image"]), 
                str(paths["sbs_image"]),
                str(paths["phenotype_positions"]), 
                str(paths["sbs_positions"]),
                str(paths["alignment_summary"]) if "alignment_summary" in existing_files else None,
                region=region,
                color_by_stitched_id=True  # Color by stitched_cell_id
            )
            fig2.suptitle(f'Modality Comparison - {plate}_{well} - Region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]')
            plt.show()
        except Exception as e:
            print(f"❌ Error comparing modality images: {e}")
    
    # 3. Show matched cells if merge results exist
    if all(f in existing_files for f in ["merged_cells", "phenotype_image", "sbs_image"]):
        print("\n--- VISUALIZING MATCHED CELLS ---")
        try:
            fig3 = visualize_matched_cells(
                str(paths["phenotype_image"]), 
                str(paths["sbs_image"]),
                str(paths["merged_cells"]), 
                region=region,
                max_distance=20.0  # Show matches up to 20px
            )
            fig3.suptitle(f'Matched Cells - {plate}_{well} - Region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]')
            plt.show()
        except Exception as e:
            print(f"❌ Error visualizing matched cells: {e}")
    
    # 4. Show individual images with higher detail if requested
    if "phenotype_image" in existing_files:
        print("\n--- DETAILED PHENOTYPE VIEW ---")
        try:
            fig4 = view_stitched_region(
                str(paths["phenotype_image"]),
                str(paths["phenotype_positions"]) if "phenotype_positions" in existing_files else None,
                region=region,
                cell_color='red',
                color_by_stitched_id=True,  # Color by stitched_cell_id
                cell_size=8,
                title=f'Phenotype Stitched Image - {plate}_{well}',
                figsize=(12, 10)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error showing phenotype image: {e}")
    
    if "sbs_image" in existing_files:
        print("\n--- DETAILED SBS VIEW ---")
        try:
            fig5 = view_stitched_region(
                str(paths["sbs_image"]),
                str(paths["sbs_positions"]) if "sbs_positions" in existing_files else None,
                region=region,
                cell_color='blue',
                color_by_stitched_id=True,  # Color by stitched_cell_id
                cell_size=8,
                title=f'SBS Stitched Image - {plate}_{well}',
                figsize=(12, 10)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error showing SBS image: {e}")
    
    # 5. Print summary statistics if available
    if "merge_summary" in existing_files:
        print("\n--- MERGE SUMMARY ---")
        try:
            summary = load_alignment_summary(str(paths["merge_summary"]))
            if 'matching_results' in summary:
                results = summary['matching_results']
                print(f"Raw matches found: {results.get('raw_matches_found', 0):,}")
                print(f"Mean match distance: {results.get('mean_match_distance', 0):.1f}px")
                print(f"Matches under 5px: {results.get('matches_under_5px', 0):,}")
                print(f"Matches under 10px: {results.get('matches_under_10px', 0):,}")
                print(f"Phenotype match rate: {results.get('match_rate_phenotype', 0):.1%}")
                print(f"SBS match rate: {results.get('match_rate_sbs', 0):.1%}")
        except Exception as e:
            print(f"❌ Error reading merge summary: {e}")
    
    print(f"\n🎉 Analysis complete for {plate}_{well}!")


def quick_well_analysis(plate: str, well: str, root_fp: str):
    """Quick analysis using the standard sampling region."""
    analyze_well_merge(plate, well, root_fp, region=(9764, 16764, 9810, 16810))


##### BREAK



def well_level_triangle_hash(positions_df: pd.DataFrame, 
                           n_triangles: int = 1000, 
                           min_distance: float = 100.0) -> pd.DataFrame:
    """Generate triangle hash from cell positions for alignment.
    
    Args:
        positions_df: DataFrame with 'i' and 'j' columns for cell positions
        n_triangles: Number of triangles to generate
        min_distance: Minimum distance between triangle vertices
        
    Returns:
        DataFrame with triangle hash information
    """
    from scipy.spatial.distance import pdist, squareform
    
    if len(positions_df) < 3:
        return pd.DataFrame()
    
    # Get coordinates
    coords = positions_df[['i', 'j']].values
    
    # Calculate all pairwise distances
    distances = squareform(pdist(coords))
    
    triangles = []
    attempts = 0
    max_attempts = n_triangles * 10
    
    while len(triangles) < n_triangles and attempts < max_attempts:
        # Randomly select 3 points
        indices = np.random.choice(len(coords), size=3, replace=False)
        
        # Check minimum distance constraint
        triangle_distances = [
            distances[indices[0], indices[1]],
            distances[indices[1], indices[2]], 
            distances[indices[2], indices[0]]
        ]
        
        if all(d > min_distance for d in triangle_distances):
            # Calculate triangle properties for hashing
            triangle_coords = coords[indices]
            
            # Sort distances to create invariant hash
            sorted_distances = sorted(triangle_distances)
            
            # Calculate triangle area using cross product
            v1 = triangle_coords[1] - triangle_coords[0]
            v2 = triangle_coords[2] - triangle_coords[0]
            area = abs(np.cross(v1, v2)) / 2
            
            # Calculate triangle centroid
            centroid = triangle_coords.mean(axis=0)
            
            triangles.append({
                'triangle_id': len(triangles),
                'vertex_ids': indices,
                'distances': sorted_distances,
                'area': area,
                'centroid_i': centroid[0],
                'centroid_j': centroid[1],
                'hash_key': f"{sorted_distances[0]:.1f}_{sorted_distances[1]:.1f}_{sorted_distances[2]:.1f}"
            })
        
        attempts += 1
    
    return pd.DataFrame(triangles)


def create_triangle_hash_lookup(triangles_df: pd.DataFrame, 
                               tolerance: float = 5.0) -> Dict[str, list]:
    """Create lookup table for triangle matching.
    
    Args:
        triangles_df: DataFrame with triangle information
        tolerance: Distance tolerance for matching
        
    Returns:
        Dictionary mapping hash keys to triangle IDs
    """
    lookup = {}
    
    for _, triangle in triangles_df.iterrows():
        hash_key = triangle['hash_key']
        triangle_id = triangle['triangle_id']
        
        # Create variations of the hash key within tolerance
        distances = triangle['distances']
        
        # Generate hash keys with small variations
        for d1_var in [-tolerance, 0, tolerance]:
            for d2_var in [-tolerance, 0, tolerance]:
                for d3_var in [-tolerance, 0, tolerance]:
                    var_distances = [
                        distances[0] + d1_var,
                        distances[1] + d2_var, 
                        distances[2] + d3_var
                    ]
                    var_key = f"{var_distances[0]:.1f}_{var_distances[1]:.1f}_{var_distances[2]:.1f}"
                    
                    if var_key not in lookup:
                        lookup[var_key] = []
                    lookup[var_key].append(triangle_id)
    
    return lookup


def match_triangles(pheno_triangles: pd.DataFrame, 
                   sbs_triangles: pd.DataFrame,
                   tolerance: float = 5.0) -> pd.DataFrame:
    """Match triangles between phenotype and SBS datasets.
    
    Args:
        pheno_triangles: Phenotype triangle DataFrame
        sbs_triangles: SBS triangle DataFrame  
        tolerance: Distance tolerance for matching
        
    Returns:
        DataFrame with matched triangles
    """
    # Create lookup table for SBS triangles
    sbs_lookup = create_triangle_hash_lookup(sbs_triangles, tolerance)
    
    matches = []
    
    for _, pheno_triangle in pheno_triangles.iterrows():
        hash_key = pheno_triangle['hash_key']
        
        # Look for matches in SBS lookup
        if hash_key in sbs_lookup:
            for sbs_triangle_id in sbs_lookup[hash_key]:
                sbs_triangle = sbs_triangles.loc[sbs_triangles['triangle_id'] == sbs_triangle_id].iloc[0]
                
                # Calculate match quality
                distance_diff = np.abs(np.array(pheno_triangle['distances']) - np.array(sbs_triangle['distances']))
                match_score = 1.0 / (1.0 + distance_diff.sum())
                
                matches.append({
                    'pheno_triangle_id': pheno_triangle['triangle_id'],
                    'sbs_triangle_id': sbs_triangle_id,
                    'match_score': match_score,
                    'pheno_centroid': (pheno_triangle['centroid_i'], pheno_triangle['centroid_j']),
                    'sbs_centroid': (sbs_triangle['centroid_i'], sbs_triangle['centroid_j'])
                })
    
    return pd.DataFrame(matches)


def estimate_transformation_from_triangles(triangle_matches: pd.DataFrame,
                                         min_matches: int = 10) -> Optional[Dict[str, Any]]:
    """Estimate coordinate transformation from triangle matches.
    
    Args:
        triangle_matches: DataFrame with matched triangles
        min_matches: Minimum number of matches required
        
    Returns:
        Dictionary with transformation parameters or None if insufficient matches
    """
    if len(triangle_matches) < min_matches:
        return None
    
    # Extract centroid coordinates
    pheno_centroids = np.array([match for match in triangle_matches['pheno_centroid']])
    sbs_centroids = np.array([match for match in triangle_matches['sbs_centroid']])
    
    # Use RANSAC-like approach to find best transformation
    best_score = 0
    best_transform = None
    
    n_iterations = min(100, len(triangle_matches) * 2)
    
    for _ in range(n_iterations):
        # Randomly sample matches
        sample_size = min(10, len(triangle_matches))
        sample_indices = np.random.choice(len(triangle_matches), size=sample_size, replace=False)
        
        sample_pheno = pheno_centroids[sample_indices]
        sample_sbs = sbs_centroids[sample_indices]
        
        # Estimate transformation using least squares
        try:
            # Center the coordinates
            pheno_mean = sample_pheno.mean(axis=0)
            sbs_mean = sample_sbs.mean(axis=0)
            
            pheno_centered = sample_pheno - pheno_mean
            sbs_centered = sample_sbs - sbs_mean
            
            # Estimate rotation and scaling using SVD
            H = pheno_centered.T @ sbs_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (determinant = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Calculate translation
            translation = sbs_mean - R @ pheno_mean
            
            # Test transformation on all matches
            transformed_pheno = (R @ pheno_centroids.T).T + translation
            distances = np.linalg.norm(transformed_pheno - sbs_centroids, axis=1)
            
            # Score based on number of inliers
            inliers = distances < 20.0  # 20 pixel tolerance
            score = inliers.sum()
            
            if score > best_score:
                best_score = score
                best_transform = {
                    'rotation_matrix': R,
                    'translation_vector': translation,
                    'score': score / len(triangle_matches),
                    'determinant': np.linalg.det(R),
                    'mean_error': distances[inliers].mean() if inliers.sum() > 0 else float('inf'),
                    'inlier_count': inliers.sum(),
                    'total_matches': len(triangle_matches)
                }
                
        except (np.linalg.LinAlgError, ValueError):
            continue
    
    return best_transform


def apply_coordinate_transformation(positions_df: pd.DataFrame,
                                  rotation_matrix: np.ndarray,
                                  translation_vector: np.ndarray) -> pd.DataFrame:
    """Apply coordinate transformation to cell positions.
    
    Args:
        positions_df: DataFrame with 'i' and 'j' columns
        rotation_matrix: 2x2 rotation matrix
        translation_vector: 2-element translation vector
        
    Returns:
        DataFrame with transformed coordinates
    """
    transformed_df = positions_df.copy()
    
    # Extract coordinates
    coords = positions_df[['i', 'j']].values
    
    # Apply transformation
    transformed_coords = (rotation_matrix @ coords.T).T + translation_vector
    
    # Update dataframe
    transformed_df['i'] = transformed_coords[:, 0]
    transformed_df['j'] = transformed_coords[:, 1]
    
    return transformed_df


def calculate_overlap_statistics(phenotype_positions: pd.DataFrame,
                               sbs_positions: pd.DataFrame) -> Dict[str, float]:
    """Calculate overlap statistics between two position datasets.
    
    Args:
        phenotype_positions: DataFrame with phenotype cell positions
        sbs_positions: DataFrame with SBS cell positions
        
    Returns:
        Dictionary with overlap statistics
    """
    # Calculate coordinate ranges
    ph_i_range = (phenotype_positions['i'].min(), phenotype_positions['i'].max())
    ph_j_range = (phenotype_positions['j'].min(), phenotype_positions['j'].max())
    
    sbs_i_range = (sbs_positions['i'].min(), sbs_positions['i'].max())
    sbs_j_range = (sbs_positions['j'].min(), sbs_positions['j'].max())
    
    # Calculate overlap in each dimension
    i_overlap = max(0, min(ph_i_range[1], sbs_i_range[1]) - max(ph_i_range[0], sbs_i_range[0]))
    j_overlap = max(0, min(ph_j_range[1], sbs_j_range[1]) - max(ph_j_range[0], sbs_j_range[0]))
    
    # Calculate total ranges
    i_total = max(ph_i_range[1], sbs_i_range[1]) - min(ph_i_range[0], sbs_i_range[0])
    j_total = max(ph_j_range[1], sbs_j_range[1]) - min(ph_j_range[0], sbs_j_range[0])
    
    # Calculate overlap fractions
    overlap_area = i_overlap * j_overlap
    total_area = i_total * j_total
    overlap_fraction = overlap_area / total_area if total_area > 0 else 0
    
    return {
        'overlap_fraction': overlap_fraction,
        'overlap_area': overlap_area,
        'total_area': total_area,
        'i_overlap': i_overlap,
        'j_overlap': j_overlap,
        'phenotype_range_i': ph_i_range,
        'phenotype_range_j': ph_j_range,
        'sbs_range_i': sbs_i_range,
        'sbs_range_j': sbs_j_range
    }


def create_alignment_summary(plate: str, well: str, 
                           alignment_results: Dict[str, Any],
                           overlap_stats: Dict[str, float],
                           triangle_stats: Dict[str, int]) -> Dict[str, Any]:
    """Create comprehensive alignment summary.
    
    Args:
        plate: Plate identifier
        well: Well identifier
        alignment_results: Results from transformation estimation
        overlap_stats: Overlap statistics
        triangle_stats: Triangle generation statistics
        
    Returns:
        Complete alignment summary dictionary
    """
    summary = {
        'plate': plate,
        'well': well,
        'status': 'success' if alignment_results is not None else 'failed',
        'scale_factor': 1.0,  # Default, should be calculated separately
        'overlap_fraction': overlap_stats.get('overlap_fraction', 0),
        'phenotype_triangles': triangle_stats.get('phenotype_triangles', 0),
        'sbs_triangles': triangle_stats.get('sbs_triangles', 0),
        'alignment': {
            'approach': 'triangle_hash',
            'transformation_type': 'rigid' if alignment_results else 'none',
            'score': alignment_results.get('score', 0) if alignment_results else 0,
            'determinant': alignment_results.get('determinant', 1) if alignment_results else 1,
            'mean_error': alignment_results.get('mean_error', float('inf')) if alignment_results else float('inf'),
            'inlier_count': alignment_results.get('inlier_count', 0) if alignment_results else 0,
            'total_matches': alignment_results.get('total_matches', 0) if alignment_results else 0,
            'region_size': 7000,  # Default region size
            'attempts': 1
        }
    }
    
    return summary


def save_alignment_results(output_dir: Path, 
                          plate: str, 
                          well: str,
                          alignment_params: pd.DataFrame,
                          alignment_summary: Dict[str, Any],
                          transformed_positions: pd.DataFrame):
    """Save alignment results to files.
    
    Args:
        output_dir: Directory to save results
        plate: Plate identifier
        well: Well identifier
        alignment_params: DataFrame with alignment parameters
        alignment_summary: Summary dictionary
        transformed_positions: Transformed phenotype positions
    """
    import yaml
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save alignment parameters
    params_path = output_dir / f"P-{plate}_W-{well}__alignment.parquet"
    alignment_params.to_parquet(params_path)
    
    # Save alignment summary
    summary_path = output_dir / f"P-{plate}_W-{well}__alignment_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(alignment_summary, f, default_flow_style=False)
    
    # Save transformed positions
    transformed_path = output_dir / f"P-{plate}_W-{well}__phenotype_transformed.parquet"
    transformed_positions.to_parquet(transformed_path)


def validate_alignment_quality(alignment_results: Dict[str, Any],
                              min_score: float = 0.5,
                              det_range: Tuple[float, float] = (0.8, 1.2)) -> bool:
    """Validate alignment quality against thresholds.
    
    Args:
        alignment_results: Results from transformation estimation
        min_score: Minimum acceptable score
        det_range: Acceptable range for transformation determinant
        
    Returns:
        True if alignment passes quality checks
    """
    if alignment_results is None:
        return False
    
    score = alignment_results.get('score', 0)
    determinant = alignment_results.get('determinant', 1)
    
    # Check score threshold
    if score < min_score:
        print(f"❌ Score {score:.3f} below threshold {min_score}")
        return False
    
    # Check determinant range (should be close to 1 for rigid transformation)
    if not (det_range[0] <= determinant <= det_range[1]):
        print(f"❌ Determinant {determinant:.3f} outside range {det_range}")
        return False
    
    print(f"✅ Alignment quality validated: score={score:.3f}, det={determinant:.3f}")
    return True


def plot_triangle_matches(pheno_triangles: pd.DataFrame,
                         sbs_triangles: pd.DataFrame, 
                         triangle_matches: pd.DataFrame,
                         figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """Plot triangle matches for visualization.
    
    Args:
        pheno_triangles: Phenotype triangles
        sbs_triangles: SBS triangles
        triangle_matches: Matched triangles
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot phenotype triangles
    if len(pheno_triangles) > 0:
        ax1.scatter(pheno_triangles['centroid_j'], pheno_triangles['centroid_i'], 
                   c='red', s=20, alpha=0.6, label=f'Phenotype ({len(pheno_triangles)})')
    ax1.set_title('Phenotype Triangles')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot SBS triangles  
    if len(sbs_triangles) > 0:
        ax2.scatter(sbs_triangles['centroid_j'], sbs_triangles['centroid_i'],
                   c='blue', s=20, alpha=0.6, label=f'SBS ({len(sbs_triangles)})')
    ax2.set_title('SBS Triangles')
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot matches
    if len(triangle_matches) > 0:
        pheno_centroids = np.array([match for match in triangle_matches['pheno_centroid']])
        sbs_centroids = np.array([match for match in triangle_matches['sbs_centroid']])
        
        ax3.scatter(pheno_centroids[:, 1], pheno_centroids[:, 0], 
                   c='red', s=30, alpha=0.7, label='Phenotype')
        ax3.scatter(sbs_centroids[:, 1], sbs_centroids[:, 0],
                   c='blue', s=30, alpha=0.7, label='SBS')
        
        # Draw connecting lines
        for i in range(len(triangle_matches)):
            ax3.plot([pheno_centroids[i, 1], sbs_centroids[i, 1]],
                    [pheno_centroids[i, 0], sbs_centroids[i, 0]],
                    'gray', alpha=0.3, linewidth=1)
    
    ax3.set_title(f'Triangle Matches ({len(triangle_matches)})')
    ax3.set_xlabel('J (pixels)')
    ax3.set_ylabel('I (pixels)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Additional utility functions that might be missing

def create_interactive_slider_figure(figsize: Tuple[int, int] = (16, 14)) -> Tuple[plt.Figure, dict]:
    """Create figure with interactive sliders for image adjustment.
    
    Args:
        figsize: Figure size
        
    Returns:
        Tuple of (figure, slider_dict)
    """
    fig = plt.figure(figsize=figsize)
    
    # Create main subplot area (leave space at bottom for sliders)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.2], hspace=0.4, bottom=0.1)
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ]
    
    # Slider area
    slider_ax1 = plt.axes([0.15, 0.05, 0.25, 0.03])
    slider_ax2 = plt.axes([0.55, 0.05, 0.25, 0.03])
    
    # Create sliders
    brightness_slider = Slider(
        slider_ax1, 'Brightness', 0.1, 2.0, valinit=1.0, valstep=0.05, valfmt='%.2f'
    )
    contrast_slider = Slider(
        slider_ax2, 'Contrast', 0.5, 3.0, valinit=1.0, valstep=0.05, valfmt='%.2f'
    )
    
    sliders = {
        'brightness': brightness_slider,
        'contrast': contrast_slider,
        'axes': axes,
        'slider_axes': [slider_ax1, slider_ax2]
    }
    
    return fig, sliders


def safe_file_load(file_path: Path, file_type: str = 'auto') -> Optional[Any]:
    """Safely load various file types with error handling.
    
    Args:
        file_path: Path to file
        file_type: Type of file ('parquet', 'yaml', 'npy', 'image', 'auto')
        
    Returns:
        Loaded data or None if failed
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        if file_type == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.parquet':
                file_type = 'parquet'
            elif suffix in ['.yaml', '.yml']:
                file_type = 'yaml'
            elif suffix == '.npy':
                file_type = 'npy'
            elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                file_type = 'image'
        
        if file_type == 'parquet':
            return pd.read_parquet(file_path)
        elif file_type == 'yaml':
            import yaml
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        elif file_type == 'npy':
            return np.load(file_path, mmap_mode='r')
        elif file_type == 'image':
            return io.imread(file_path)
        else:
            print(f"❌ Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def memory_efficient_image_crop(image_path: Path, 
                               region: Tuple[int, int, int, int],
                               downsample: int = 1) -> Optional[np.ndarray]:
    """Memory-efficiently crop a region from a large image file.
    
    Args:
        image_path: Path to image file
        region: (i_min, i_max, j_min, j_max) crop region
        downsample: Downsampling factor
        
    Returns:
        Cropped image array or None if failed
    """
    try:
        # Memory map the image
        image = np.load(image_path, mmap_mode='r')
        
        i_min, i_max, j_min, j_max = region
        
        # Validate bounds
        i_min = max(0, min(i_min, image.shape[0]))
        i_max = max(i_min, min(i_max, image.shape[0]))
        j_min = max(0, min(j_min, image.shape[1]))
        j_max = max(j_min, min(j_max, image.shape[1]))
        
        # Extract region with optional downsampling
        if downsample > 1:
            cropped = np.array(image[i_min:i_max:downsample, j_min:j_max:downsample])
        else:
            cropped = np.array(image[i_min:i_max, j_min:j_max])
        
        return cropped
        
    except Exception as e:
        print(f"❌ Error cropping image: {e}")
        return None
    

"""Helper functions for evaluating results of merge process - Fixed Version."""

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

# Import skimage for image loading
try:
    from skimage import io
except ImportError:
    print("Warning: skimage not available, some image loading functions may not work")
    io = None

from lib.shared.eval import plot_plate_heatmap
from lib.shared.configuration_utils import plot_merge_example
from lib.merge.merge import build_linear_model
from lib.merge.well_alignment import (
    sample_region_for_alignment,
    calculate_scale_factor_from_positions,
    scale_coordinates,
)


def load_stitched_image(image_path: str, downsample: int = 1) -> np.ndarray:
    """Load a stitched image from .npy file with optional downsampling.
    
    Args:
        image_path: Path to the .npy stitched image file
        downsample: Downsampling factor (1 = no downsampling, 2 = half size, etc.)
        
    Returns:
        Loaded and optionally downsampled image array
    """
    image = np.load(image_path)
    
    if downsample > 1:
        if image.ndim == 2:
            # 2D image
            image = image[::downsample, ::downsample]
        elif image.ndim == 3:
            # 3D image (channels last)
            image = image[::downsample, ::downsample, :]
        print(f"Downsampled by factor {downsample}, new shape: {image.shape}")
    
    return image


def load_cell_positions(positions_path: str) -> pd.DataFrame:
    """Load cell positions from parquet file.
    
    Args:
        positions_path: Path to parquet file with cell positions
        
    Returns:
        DataFrame with cell positions and metadata
    """
    return pd.read_parquet(positions_path)


def load_alignment_summary(summary_path: str) -> Dict[str, Any]:
    """Load alignment summary from YAML file.
    
    Args:
        summary_path: Path to alignment summary YAML file
        
    Returns:
        Dictionary with alignment parameters and statistics
    """
    with open(summary_path, 'r') as f:
        return yaml.safe_load(f)


def normalize_image_for_display(image: np.ndarray, 
                              percentile_clip: Tuple[float, float] = (1, 99),
                              gamma: float = 1.0) -> np.ndarray:
    """Normalize image for better visualization.
    
    Args:
        image: Raw image array
        percentile_clip: Lower and upper percentiles for clipping
        gamma: Gamma correction factor
        
    Returns:
        Normalized image ready for display
    """
    # Handle different image dimensions
    if image.ndim == 3:
        # Multi-channel image - use first channel or create RGB
        if image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.shape[2] > 3:
            # Take first 3 channels for RGB
            image = image[:, :, :3]
    
    # Clip extreme values
    lower, upper = np.percentile(image, percentile_clip)
    image = np.clip(image, lower, upper)
    
    # Normalize to 0-1
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Apply gamma correction
    if gamma != 1.0:
        image = np.power(image, gamma)
    
    return image


def view_stitched_region(image_path: str,
                        positions_path: Optional[str] = None,
                        region: Optional[Tuple[int, int, int, int]] = None,
                        cell_color: str = 'red',
                        color_by_stitched_id: bool = False,
                        cell_size: int = 5,
                        downsample: int = 4,
                        title: str = "Stitched Image Region",
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """View a region of a stitched image with optional cell overlay.
    
    Args:
        image_path: Path to stitched image .npy file
        positions_path: Optional path to cell positions parquet file
        region: Tuple of (i_min, i_max, j_min, j_max) for cropping
        cell_color: Color for cell position markers (ignored if color_by_stitched_id=True)
        color_by_stitched_id: If True, color cells by their stitched_cell_id
        cell_size: Size of cell markers
        downsample: Downsampling factor for the image (default 4)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Load image with downsampling
    image = load_stitched_image(image_path, downsample=downsample)
    print(f"Loaded image shape: {image.shape}")
    
    # Normalize for display
    display_image = normalize_image_for_display(image)
    
    # Adjust region for downsampling
    if region is not None:
        i_min, i_max, j_min, j_max = region
        # Scale region coordinates by downsample factor
        i_min_ds, i_max_ds = i_min // downsample, i_max // downsample
        j_min_ds, j_max_ds = j_min // downsample, j_max // downsample
        display_image = display_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        print(f"Cropped to region: {region} (downsampled: [{i_min_ds}, {i_max_ds}, {j_min_ds}, {j_max_ds}])")
        # Use original coordinates for extent
        extent = [j_min, j_max, i_max, i_min]
    else:
        region = (0, image.shape[0] * downsample, 0, image.shape[1] * downsample)
        i_min, i_max, j_min, j_max = region
        extent = [j_min, j_max, i_max, i_min]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show image
    if display_image.ndim == 3:
        ax.imshow(display_image, extent=extent)
    else:
        ax.imshow(display_image, cmap='gray', extent=extent)
    
    # Overlay cell positions if provided
    if positions_path is not None:
        positions = load_cell_positions(positions_path)
        print(f"Loaded {len(positions)} cell positions")
        
        # Filter to region
        region_positions = positions[
            (positions['i'] >= i_min) & (positions['i'] <= i_max) &
            (positions['j'] >= j_min) & (positions['j'] <= j_max)
        ]
        print(f"Found {len(region_positions)} cells in region")
        
        if len(region_positions) > 0:
            if color_by_stitched_id and 'stitched_cell_id' in region_positions.columns:
                # Color by stitched_cell_id
                unique_ids = region_positions['stitched_cell_id'].unique()
                
                scatter = ax.scatter(region_positions['j'], region_positions['i'], 
                          c=region_positions['stitched_cell_id'], s=cell_size, 
                          alpha=0.7, edgecolors='white', linewidths=0.5,
                          cmap='tab20')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Stitched Cell ID')
                
                print(f"Colored by stitched_cell_id: {len(unique_ids)} unique IDs in region")
            else:
                # Use single color
                ax.scatter(region_positions['j'], region_positions['i'], 
                          c=cell_color, s=cell_size, alpha=0.7, edgecolors='white', linewidths=0.5)
                
                if color_by_stitched_id:
                    print("⚠️  stitched_cell_id column not found, using single color")
    
    ax.set_title(title)
    ax.set_xlabel('J (pixels)')
    ax.set_ylabel('I (pixels)')
    
    return fig


def compare_modality_images(phenotype_image_path: str,
                           sbs_image_path: str,
                           phenotype_positions_path: str,
                           sbs_positions_path: str,
                           alignment_summary_path: Optional[str] = None,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           color_by_stitched_id: bool = False,
                           figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
    """Compare stitched images from both modalities side by side.
    
    Args:
        phenotype_image_path: Path to phenotype stitched image
        sbs_image_path: Path to SBS stitched image  
        phenotype_positions_path: Path to phenotype cell positions
        sbs_positions_path: Path to SBS cell positions
        alignment_summary_path: Optional path to alignment summary
        region: Region to crop both images to
        color_by_stitched_id: If True, color cells by their stitched_cell_id
        figsize: Figure size
        
    Returns:
        Matplotlib figure with side-by-side comparison
    """
    # Load images
    pheno_image = load_stitched_image(phenotype_image_path)
    sbs_image = load_stitched_image(sbs_image_path)
    
    # Load positions
    pheno_positions = load_cell_positions(phenotype_positions_path)
    sbs_positions = load_cell_positions(sbs_positions_path)
    
    print(f"Phenotype: {pheno_image.shape} image, {len(pheno_positions)} cells")
    print(f"SBS: {sbs_image.shape} image, {len(sbs_positions)} cells")
    
    # Load alignment info if available
    alignment_info = ""
    if alignment_summary_path is not None and Path(alignment_summary_path).exists():
        summary = load_alignment_summary(alignment_summary_path)
        if 'alignment' in summary:
            align_data = summary['alignment']
            alignment_info = f"Score: {align_data.get('score', 0):.3f}, Det: {align_data.get('determinant', 1):.3f}"
    
    # Normalize images
    pheno_display = normalize_image_for_display(pheno_image)
    sbs_display = normalize_image_for_display(sbs_image)
    
    # Apply region cropping if specified
    if region is not None:
        i_min, i_max, j_min, j_max = region
        pheno_display = pheno_display[i_min:i_max, j_min:j_max]
        sbs_display = sbs_display[i_min:i_max, j_min:j_max]
        
        # Filter positions to region
        pheno_positions = pheno_positions[
            (pheno_positions['i'] >= i_min) & (pheno_positions['i'] <= i_max) &
            (pheno_positions['j'] >= j_min) & (pheno_positions['j'] <= j_max)
        ]
        sbs_positions = sbs_positions[
            (sbs_positions['i'] >= i_min) & (sbs_positions['i'] <= i_max) &
            (sbs_positions['j'] >= j_min) & (sbs_positions['j'] <= j_max)
        ]
    else:
        region = (0, min(pheno_image.shape[0], sbs_image.shape[0]), 
                 0, min(pheno_image.shape[1], sbs_image.shape[1]))
        i_min, i_max, j_min, j_max = region
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Phenotype image
    if pheno_display.ndim == 3:
        ax1.imshow(pheno_display, extent=[j_min, j_max, i_max, i_min])
    else:
        ax1.imshow(pheno_display, cmap='gray', extent=[j_min, j_max, i_max, i_min])
    
    if color_by_stitched_id and 'stitched_cell_id' in pheno_positions.columns:
        scatter1 = ax1.scatter(pheno_positions['j'], pheno_positions['i'], 
                   c=pheno_positions['stitched_cell_id'], s=3, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, cmap='tab20')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Phenotype Stitched Cell ID')
    else:
        ax1.scatter(pheno_positions['j'], pheno_positions['i'], 
                   c='red', s=3, alpha=0.8, edgecolors='white', linewidths=0.3)
    
    ax1.set_title(f'Phenotype\n{len(pheno_positions)} cells in region')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    
    # SBS image
    if sbs_display.ndim == 3:
        ax2.imshow(sbs_display, extent=[j_min, j_max, i_max, i_min])
    else:
        ax2.imshow(sbs_display, cmap='gray', extent=[j_min, j_max, i_max, i_min])
    
    if color_by_stitched_id and 'stitched_cell_id' in sbs_positions.columns:
        scatter2 = ax2.scatter(sbs_positions['j'], sbs_positions['i'], 
                   c=sbs_positions['stitched_cell_id'], s=3, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, cmap='tab20')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('SBS Stitched Cell ID')
    else:
        ax2.scatter(sbs_positions['j'], sbs_positions['i'], 
                   c='blue', s=3, alpha=0.8, edgecolors='white', linewidths=0.3)
    
    ax2.set_title(f'SBS\n{len(sbs_positions)} cells in region')
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    
    if alignment_info:
        fig.suptitle(f'Modality Comparison - {alignment_info}')
    else:
        fig.suptitle('Modality Comparison')
    
    plt.tight_layout()
    return fig


def visualize_matched_cells(phenotype_image_path: str,
                           sbs_image_path: str,
                           matched_cells_path: str,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           max_distance: float = 10.0,
                           downsample: int = 4,
                           figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
    """Visualize matched cells overlaid on both modality images.
    
    Args:
        phenotype_image_path: Path to phenotype stitched image
        sbs_image_path: Path to SBS stitched image
        matched_cells_path: Path to matched cells parquet file
        region: Region to focus on
        max_distance: Maximum distance to show matches
        downsample: Downsampling factor for images (default 4)
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing matches
    """
    # Load images with downsampling
    pheno_image = normalize_image_for_display(load_stitched_image(phenotype_image_path, downsample=downsample))
    sbs_image = normalize_image_for_display(load_stitched_image(sbs_image_path, downsample=downsample))
    
    # Load matched cells
    matches = pd.read_parquet(matched_cells_path)
    print(f"Loaded {len(matches)} matched cells")
    
    # Filter by distance if specified
    if max_distance is not None:
        matches = matches[matches['distance'] <= max_distance]
        print(f"Filtered to {len(matches)} matches within {max_distance}px")
    
    # Apply region if specified
    if region is not None:
        i_min, i_max, j_min, j_max = region
        # Scale region for downsampled images
        i_min_ds, i_max_ds = i_min // downsample, i_max // downsample
        j_min_ds, j_max_ds = j_min // downsample, j_max // downsample
        
        pheno_image = pheno_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        sbs_image = sbs_image[i_min_ds:i_max_ds, j_min_ds:j_max_ds]
        
        # Filter matches to region (using original coordinates)
        matches = matches[
            (matches['i_0'] >= i_min) & (matches['i_0'] <= i_max) &
            (matches['j_0'] >= j_min) & (matches['j_0'] <= j_max) &
            (matches['i_1'] >= i_min) & (matches['i_1'] <= i_max) &
            (matches['j_1'] >= j_min) & (matches['j_1'] <= j_max)
        ]
        print(f"Found {len(matches)} matches in region")
        extent = [j_min, j_max, i_max, i_min]
    else:
        region = (0, min(pheno_image.shape[0], sbs_image.shape[0]) * downsample, 
                 0, min(pheno_image.shape[1], sbs_image.shape[1]) * downsample)
        i_min, i_max, j_min, j_max = region
        extent = [j_min, j_max, i_max, i_min]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Show images
    if pheno_image.ndim == 3:
        ax1.imshow(pheno_image, extent=extent)
    else:
        ax1.imshow(pheno_image, cmap='gray', extent=extent)
    
    if sbs_image.ndim == 3:
        ax2.imshow(sbs_image, extent=extent)
    else:
        ax2.imshow(sbs_image, cmap='gray', extent=extent)
    
    # Plot matches
    if len(matches) > 0:
        # Color code by distance
        distances = matches['distance']
        scatter1 = ax1.scatter(matches['j_0'], matches['i_0'], 
                              c=distances, s=20, cmap='viridis', alpha=0.8,
                              edgecolors='white', linewidths=0.5)
        scatter2 = ax2.scatter(matches['j_1'], matches['i_1'], 
                              c=distances, s=20, cmap='viridis', alpha=0.8,
                              edgecolors='white', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label('Match Distance (pixels)')
    
    ax1.set_title(f'Phenotype Matches\n{len(matches)} cells')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    
    ax2.set_title(f'SBS Matches\n{len(matches)} cells')
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    
    if len(matches) > 0:
        mean_dist = matches['distance'].mean()
        fig.suptitle(f'Matched Cells (Mean Distance: {mean_dist:.1f}px)')
    else:
        fig.suptitle('Matched Cells (No matches in region)')
    
    plt.tight_layout()
    return fig


def plot_alignment_overview(phenotype_positions_path: str,
                           sbs_positions_path: str,
                           transformed_positions_path: Optional[str] = None,
                           alignment_summary_path: Optional[str] = None,
                           sample_size: int = 5000,
                           figsize: Tuple[int, int] = (18, 6)) -> plt.Figure:
    """Plot overview of coordinate alignment process.
    
    Args:
        phenotype_positions_path: Path to original phenotype positions
        sbs_positions_path: Path to SBS positions
        transformed_positions_path: Path to transformed phenotype positions
        alignment_summary_path: Path to alignment summary
        sample_size: Number of cells to sample for plotting
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing alignment progression
    """
    # Load data
    pheno_pos = load_cell_positions(phenotype_positions_path)
    sbs_pos = load_cell_positions(sbs_positions_path)
    
    print(f"Loaded {len(pheno_pos)} phenotype and {len(sbs_pos)} SBS positions")
    
    # Sample for plotting
    if len(pheno_pos) > sample_size:
        pheno_pos = pheno_pos.sample(n=sample_size)
    if len(sbs_pos) > sample_size:
        sbs_pos = sbs_pos.sample(n=sample_size)
    
    # Load alignment summary if available
    alignment_info = {}
    if alignment_summary_path and Path(alignment_summary_path).exists():
        summary = load_alignment_summary(alignment_summary_path)
        alignment_info = summary.get('alignment', {})
    
    # Determine number of subplots
    n_plots = 2
    if transformed_positions_path and Path(transformed_positions_path).exists():
        n_plots = 3
        transformed_pos = load_cell_positions(transformed_positions_path)
        if len(transformed_pos) > sample_size:
            transformed_pos = transformed_pos.sample(n=sample_size)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # Plot 1: Original coordinates
    ax1.scatter(pheno_pos['j'], pheno_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype')
    ax1.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
    ax1.set_title('Original Coordinates')
    ax1.set_xlabel('J (pixels)')
    ax1.set_ylabel('I (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and show coordinate ranges
    pheno_range_i = (pheno_pos['i'].min(), pheno_pos['i'].max())
    pheno_range_j = (pheno_pos['j'].min(), pheno_pos['j'].max())
    sbs_range_i = (sbs_pos['i'].min(), sbs_pos['i'].max())
    sbs_range_j = (sbs_pos['j'].min(), sbs_pos['j'].max())
    
    ax1.text(0.02, 0.98, f'Pheno I: {pheno_range_i[0]:.0f}-{pheno_range_i[1]:.0f}\n'
                         f'Pheno J: {pheno_range_j[0]:.0f}-{pheno_range_j[1]:.0f}\n'
                         f'SBS I: {sbs_range_i[0]:.0f}-{sbs_range_i[1]:.0f}\n'
                         f'SBS J: {sbs_range_j[0]:.0f}-{sbs_range_j[1]:.0f}',
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: After scaling (use transformed if available, otherwise estimate scaling)
    if 'transformed_pos' in locals():
        scaled_pos = transformed_pos
        title2 = 'After Scaling & Transform'
    else:
        # Estimate scaling for visualization
        scale_factor = alignment_info.get('scale_factor', 1.0)
        scaled_pos = pheno_pos.copy()
        scaled_pos['i'] = scaled_pos['i'] * scale_factor
        scaled_pos['j'] = scaled_pos['j'] * scale_factor
        title2 = f'After Scaling (factor: {scale_factor:.3f})'
    
    ax2.scatter(scaled_pos['j'], scaled_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype (scaled)')
    ax2.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
    ax2.set_title(title2)
    ax2.set_xlabel('J (pixels)')
    ax2.set_ylabel('I (pixels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add alignment info if available
    if alignment_info:
        info_text = f"Score: {alignment_info.get('score', 0):.3f}\n"
        info_text += f"Det: {alignment_info.get('determinant', 1):.3f}\n"
        info_text += f"Type: {alignment_info.get('transformation_type', 'unknown')}"
        
        ax2.text(0.02, 0.98, info_text,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Final transformation (if available)
    if n_plots == 3 and 'transformed_pos' in locals():
        ax3.scatter(transformed_pos['j'], transformed_pos['i'], c='red', s=1, alpha=0.6, label='Phenotype (final)')
        ax3.scatter(sbs_pos['j'], sbs_pos['i'], c='blue', s=1, alpha=0.6, label='SBS')
        ax3.set_title('After Full Transformation')
        ax3.set_xlabel('J (pixels)')
        ax3.set_ylabel('I (pixels)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calculate overlap
        trans_range_i = (transformed_pos['i'].min(), transformed_pos['i'].max())
        trans_range_j = (transformed_pos['j'].min(), transformed_pos['j'].max())
        
        overlap_i = max(0, min(trans_range_i[1], sbs_range_i[1]) - max(trans_range_i[0], sbs_range_i[0]))
        overlap_j = max(0, min(trans_range_j[1], sbs_range_j[1]) - max(trans_range_j[0], sbs_range_j[0]))
        
        total_i = max(trans_range_i[1], sbs_range_i[1]) - min(trans_range_i[0], sbs_range_i[0])
        total_j = max(trans_range_j[1], sbs_range_j[1]) - min(trans_range_j[0], sbs_range_j[0])
        
        overlap_fraction = (overlap_i * overlap_j) / (total_i * total_j) if total_i > 0 and total_j > 0 else 0
        
        ax3.text(0.02, 0.98, f'Overlap: {overlap_fraction:.1%}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig


def get_merge_file_paths(plate: str, well: str, root_fp: str) -> Dict[str, Path]:
    """Get file paths using the same structure as the pipeline.
    
    Matches the approach used in run_well_alignment_qc and other pipeline functions.
    
    Args:
        plate: Plate identifier
        well: Well identifier
        root_fp: Root file path for the project
        
    Returns:
        Dictionary with all relevant file paths
    """
    try:
        from lib.shared.file_utils import get_filename
        
        root_path = Path(root_fp)
        merge_fp = root_path / "merge"
        
        # Build wildcards dict for filename generation
        wildcards = {"plate": plate, "well": well}
        
        paths = {
            # Stitched images
            "phenotype_image": merge_fp / "stitched_images" / get_filename(wildcards, "phenotype_stitched_image", "npy"),
            "sbs_image": merge_fp / "stitched_images" / get_filename(wildcards, "sbs_stitched_image", "npy"),
            
            # Stitched masks
            "phenotype_mask": merge_fp / "stitched_masks" / get_filename(wildcards, "phenotype_stitched_mask", "npy"),
            "sbs_mask": merge_fp / "stitched_masks" / get_filename(wildcards, "sbs_stitched_mask", "npy"),
            
            # Cell positions
            "phenotype_positions": merge_fp / "cell_positions" / get_filename(wildcards, "phenotype_cell_positions", "parquet"),
            "sbs_positions": merge_fp / "cell_positions" / get_filename(wildcards, "sbs_cell_positions", "parquet"),
            
            # Well alignment outputs
            "phenotype_scaled": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_scaled", "parquet"),
            "phenotype_triangles": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_triangles", "parquet"),
            "sbs_triangles": merge_fp / "well_alignment" / get_filename(wildcards, "sbs_triangles", "parquet"),
            "alignment_params": merge_fp / "well_alignment" / get_filename(wildcards, "alignment", "parquet"),
            "alignment_summary": merge_fp / "well_alignment" / get_filename(wildcards, "alignment_summary", "yaml"),
            "phenotype_transformed": merge_fp / "well_alignment" / get_filename(wildcards, "phenotype_transformed", "parquet"),
            
            # Well cell merge outputs  
            "raw_matches": merge_fp / "well_cell_merge" / get_filename(wildcards, "raw_matches", "parquet"),
            "merged_cells": merge_fp / "well_cell_merge" / get_filename(wildcards, "merged_cells", "parquet"),
            "merge_summary": merge_fp / "well_cell_merge" / get_filename(wildcards, "merge_summary", "yaml"),
            
            # Deduplication outputs
            "deduplicated_cells": merge_fp / "well_merge_deduplicate" / get_filename(wildcards, "deduplicated_cells", "parquet"),
            "dedup_summary": merge_fp / "well_merge_deduplicate" / get_filename(wildcards, "dedup_summary", "yaml"),
        }
        
    except ImportError:
        # Fallback to manual path construction if file_utils not available
        print("Warning: lib.shared.file_utils not available, using manual path construction")
        
        root_path = Path(root_fp)
        merge_fp = root_path / "merge"
        prefix = f"P-{plate}_W-{well}__"
        
        paths = {
            # Stitched images
            "phenotype_image": merge_fp / "stitched_images" / f"{prefix}phenotype_stitched_image.npy",
            "sbs_image": merge_fp / "stitched_images" / f"{prefix}sbs_stitched_image.npy",
            
            # Stitched masks
            "phenotype_mask": merge_fp / "stitched_masks" / f"{prefix}phenotype_stitched_mask.npy",
            "sbs_mask": merge_fp / "stitched_masks" / f"{prefix}sbs_stitched_mask.npy",
            
            # Cell positions
            "phenotype_positions": merge_fp / "cell_positions" / f"{prefix}phenotype_cell_positions.parquet",
            "sbs_positions": merge_fp / "cell_positions" / f"{prefix}sbs_cell_positions.parquet",
            
            # Well alignment outputs
            "phenotype_scaled": merge_fp / "well_alignment" / f"{prefix}phenotype_scaled.parquet",
            "phenotype_triangles": merge_fp / "well_alignment" / f"{prefix}phenotype_triangles.parquet",
            "sbs_triangles": merge_fp / "well_alignment" / f"{prefix}sbs_triangles.parquet",
            "alignment_params": merge_fp / "well_alignment" / f"{prefix}alignment.parquet",
            "alignment_summary": merge_fp / "well_alignment" / f"{prefix}alignment_summary.yaml",
            "phenotype_transformed": merge_fp / "well_alignment" / f"{prefix}phenotype_transformed.parquet",
            
            # Well cell merge outputs  
            "raw_matches": merge_fp / "well_cell_merge" / f"{prefix}raw_matches.parquet",
            "merged_cells": merge_fp / "well_cell_merge" / f"{prefix}merged_cells.parquet",
            "merge_summary": merge_fp / "well_cell_merge" / f"{prefix}merge_summary.yaml",
            
            # Deduplication outputs
            "deduplicated_cells": merge_fp / "well_merge_deduplicate" / f"{prefix}deduplicated_cells.parquet",
            "dedup_summary": merge_fp / "well_merge_deduplicate" / f"{prefix}dedup_summary.yaml",
        }
    
    return paths


def analyze_well_merge(plate: str, well: str, root_fp: str, region: Optional[Tuple[int, int, int, int]] = None):
    """Comprehensive analysis of a well merge result using pipeline file structure.
    
    Args:
        plate: Plate identifier
        well: Well identifier  
        root_fp: Root file path for the project (same as used in pipeline)
        region: Optional region to focus analysis on (i_min, i_max, j_min, j_max)
    """
    print(f"=== ANALYZING WELL MERGE: {plate}_{well} ===")
    
    # Get file paths using pipeline structure
    paths = get_merge_file_paths(plate, well, root_fp)
    
    # Check which files exist
    existing_files = []
    for name, path in paths.items():
        if path.exists():
            existing_files.append(name)
            print(f"✅ Found: {name}")
        else:
            print(f"❌ Missing: {name}")
    
    if not existing_files:
        print("❌ No merge files found! Check your paths.")
        return
    
    # Use the specified sampling region if provided, otherwise use default
    if region is None:
        region = (9764, 16764, 9810, 16810)  # Your specified sampling region
        print(f"Using default sampling region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]")
    else:
        print(f"Using custom region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]")
    
    # 1. Show alignment overview if position files exist
    if all(f in existing_files for f in ["phenotype_positions", "sbs_positions"]):
        print("\n--- PLOTTING ALIGNMENT OVERVIEW ---")
        try:
            fig1 = plot_alignment_overview(
                str(paths["phenotype_positions"]), 
                str(paths["sbs_positions"]),
                str(paths["phenotype_transformed"]) if "phenotype_transformed" in existing_files else None,
                str(paths["alignment_summary"]) if "alignment_summary" in existing_files else None
            )
            fig1.suptitle(f'Alignment Overview - {plate}_{well}')
            plt.show()
        except Exception as e:
            print(f"❌ Error plotting alignment overview: {e}")
    
    # 2. Compare modality images if they exist
    if all(f in existing_files for f in ["phenotype_image", "sbs_image", "phenotype_positions", "sbs_positions"]):
        print("\n--- COMPARING MODALITY IMAGES ---")
        try:
            fig2 = compare_modality_images(
                str(paths["phenotype_image"]), 
                str(paths["sbs_image"]),
                str(paths["phenotype_positions"]), 
                str(paths["sbs_positions"]),
                str(paths["alignment_summary"]) if "alignment_summary" in existing_files else None,
                region=region,
                color_by_stitched_id=True  # Color by stitched_cell_id
            )
            fig2.suptitle(f'Modality Comparison - {plate}_{well} - Region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]')
            plt.show()
        except Exception as e:
            print(f"❌ Error comparing modality images: {e}")
    
    # 3. Show matched cells if merge results exist
    if all(f in existing_files for f in ["merged_cells", "phenotype_image", "sbs_image"]):
        print("\n--- VISUALIZING MATCHED CELLS ---")
        try:
            fig3 = visualize_matched_cells(
                str(paths["phenotype_image"]), 
                str(paths["sbs_image"]),
                str(paths["merged_cells"]), 
                region=region,
                max_distance=20.0  # Show matches up to 20px
            )
            fig3.suptitle(f'Matched Cells - {plate}_{well} - Region: i=[{region[0]}, {region[1]}], j=[{region[2]}, {region[3]}]')
            plt.show()
        except Exception as e:
            print(f"❌ Error visualizing matched cells: {e}")
    
    # 4. Show individual images with higher detail if requested
    if "phenotype_image" in existing_files:
        print("\n--- DETAILED PHENOTYPE VIEW ---")
        try:
            fig4 = view_stitched_region(
                str(paths["phenotype_image"]),
                str(paths["phenotype_positions"]) if "phenotype_positions" in existing_files else None,
                region=region,
                cell_color='red',
                color_by_stitched_id=True,  # Color by stitched_cell_id
                cell_size=8,
                title=f'Phenotype Stitched Image - {plate}_{well}',
                figsize=(12, 10)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error showing phenotype image: {e}")
    
    if "sbs_image" in existing_files:
        print("\n--- DETAILED SBS VIEW ---")
        try:
            fig5 = view_stitched_region(
                str(paths["sbs_image"]),
                str(paths["sbs_positions"]) if "sbs_positions" in existing_files else None,
                region=region,
                cell_color='blue',
                color_by_stitched_id=True,  # Color by stitched_cell_id
                cell_size=8,
                title=f'SBS Stitched Image - {plate}_{well}',
                figsize=(12, 10)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error showing SBS image: {e}")
    
    # 5. Print summary statistics if available
    if "merge_summary" in existing_files:
        print("\n--- MERGE SUMMARY ---")
        try:
            summary = load_alignment_summary(str(paths["merge_summary"]))
            if 'matching_results' in summary:
                results = summary['matching_results']
                print(f"Raw matches found: {results.get('raw_matches_found', 0):,}")
                print(f"Mean match distance: {results.get('mean_match_distance', 0):.1f}px")
                print(f"Matches under 5px: {results.get('matches_under_5px', 0):,}")
                print(f"Matches under 10px: {results.get('matches_under_10px', 0):,}")
                print(f"Phenotype match rate: {results.get('match_rate_phenotype', 0):.1%}")
                print(f"SBS match rate: {results.get('match_rate_sbs', 0):.1%}")
        except Exception as e:
            print(f"❌ Error reading merge summary: {e}")
    
    print(f"\n🎉 Analysis complete for {plate}_{well}!")


def quick_well_analysis(plate: str, well: str, root_fp: str):
    """Quick analysis using the standard sampling region."""
    analyze_well_merge(plate, well, root_fp, region=(9764, 16764, 9810, 16810))


# Now include all the original functions that were cut off in the formatting

def plot_sbs_ph_matching_heatmap(
    df_merge,
    df_info,
    target="sbs",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plots the rate of matching segmented cells between phenotype and SBS datasets by well and tile in a convenient plate layout.

    Args:
        df_merge: DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of merge_triangle_hash. Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to SBS data.
        df_info: DataFrame of all cells segmented from either phenotype or SBS images, e.g., concatenated outputs for all tiles
            and wells of extract_phenotype_minimal(data_phenotype=nulcei, nuclei=nuclei), often used as `sbs_cell_info`
            rule in Snakemake.
        target: Which dataset to use as the target, e.g., if target='sbs', plots the fraction of cells in each SBS tile
            that match to a phenotype cell. Should match the information stored in df_info; if df_info is a table of all
            segmented cells from SBS tiles, then target should be set as 'sbs'.
        shape: Shape of subplot for each well used in `plot_plate_heatmap`. Defaults to 'square' and infers shape based on
            the value of `target`.
        plate: Plate type for `plot_plate_heatmap`, options are {'6W', '24W', '96W'}.
        return_plot: If True, returns `df_summary`.
        return_summary: If True, returns `df_summary`.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        df_summary: DataFrame used for plotting, returned if `return_summary=True`.
        axes: Numpy array of matplotlib Axes objects.
    """
    # Determine the merge columns and source based on the target
    if target == "sbs":
        merge_cols = ["site", "cell_1"]
        source = "phenotype"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_sbs"
    elif target == "phenotype":
        merge_cols = ["tile", "cell_0"]
        source = "sbs"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_ph"
    else:
        raise ValueError("target = {} not implemented".format(target))

    # Calculate the summary dataframe
    df_summary = (
        df_info.rename(columns={"tile": merge_cols[0], "cell": merge_cols[1]})[
            ["well"] + merge_cols
        ]
        .merge(
            df_merge[["well"] + merge_cols + ["distance"]],
            how="left",
            on=["well"] + merge_cols,
        )
        .assign(matched=lambda x: x["distance"].notna())
        .groupby(["well"] + merge_cols[:1])["matched"]
        .value_counts(normalize=True)
        .rename("fraction of {} cells matched to {} cells".format(target, source))
        .to_frame()
        .reset_index()
        .query("matched==True")
        .drop(columns="matched")
        .rename(columns={merge_cols[0]: "tile"})
    )

    if return_summary and return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes[0]
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return axes[0]
    elif return_summary:
        return df_summary
    else:
        return None


def plot_channel_histogram(df_before, df_after, channel_min_cutoff=0):
    """Generates a histogram of channel values with raw counts and consistent bin edges.

    Args:
        df_before: DataFrame containing channel values before cleaning.
        df_after: DataFrame containing channel values after cleaning.
        channel_min_cutoff: Threshold value to mark with a red vertical line. Defaults to 0.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))

    # Calculate bin edges based on the full range of data
    min_val = min(df_before["channels_min"].min(), df_after["channels_min"].min())
    max_val = max(df_before["channels_min"].max(), df_after["channels_min"].max())
    bins = np.linspace(min_val, max_val, 201)  # 201 edges make 200 bins

    # Plot histograms with raw counts instead of density
    plt.hist(
        df_before["channels_min"].dropna(),
        bins=bins,
        color="blue",
        alpha=0.5,
        label="Before clean",
    )
    plt.hist(
        df_after["channels_min"].dropna(),
        bins=bins,
        color="orange",
        alpha=0.5,
        label="After clean",
    )

    # Add vertical line for channel_min_cutoff
    plt.axvline(channel_min_cutoff, color="red", linestyle="--", label="Cutoff")

    plt.title("Histogram of channels_min Values")
    plt.xlabel("channels_min")
    plt.ylabel("Count")
    plt.legend()
    return fig


def plot_cell_positions(df_merge, title, color=None, hue="channels_min"):
    """Generates a scatter plot of cell positions in the i_0, j_0 coordinate space.

    Args:
        df_merge: DataFrame containing cell position data with i_0, j_0 columns.
        title: Plot title.
        color: Fixed color for all points. If specified, overrides hue.
        hue: Column name for color variation. Defaults to 'channels_min'.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 20))

    # Plot scatter with either fixed color or hue-based coloring
    if color is not None:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", color=color, alpha=0.5)
    else:
        sns.scatterplot(data=df_merge, x="i_0", y="j_0", hue=hue, alpha=0.5)

    plt.title(title)
    plt.xlabel("i_0")
    plt.ylabel("j_0")
    return fig


# Additional utility functions for completeness

def batch_qc_report(base_path, plate_wells):
    """Generate QC reports for multiple wells.

    Parameters:
    -----------
    base_path : str
        Path to analysis outputs
    plate_wells : list of tuples
        [(plate1, well1), (plate2, well2), ...]
    """
    print("BATCH QC REPORT")
    print("=" * 60)

    summary = []

    for plate, well in plate_wells:
        # Use the analyze_well_merge function instead of StitchQC for simplicity
        try:
            paths = get_merge_file_paths(str(plate), well, base_path)
            
            # Quick file check
            ph_exists = paths["phenotype_positions"].exists()
            sbs_exists = paths["sbs_positions"].exists()

            ph_count = 0
            sbs_count = 0

            if ph_exists:
                ph_count = len(pd.read_parquet(paths["phenotype_positions"]))
            if sbs_exists:
                sbs_count = len(pd.read_parquet(paths["sbs_positions"]))

            summary.append(
                {
                    "plate": plate,
                    "well": well,
                    "ph_exists": ph_exists,
                    "sbs_exists": sbs_exists,
                    "ph_cells": ph_count,
                    "sbs_cells": sbs_count,
                    "status": "OK"
                    if ph_exists and sbs_exists and ph_count > 0 and sbs_count > 0
                    else "ISSUE",
                }
            )
        except Exception as e:
            print(f"Error processing {plate}_{well}: {e}")
            summary.append(
                {
                    "plate": plate,
                    "well": well,
                    "ph_exists": False,
                    "sbs_exists": False,
                    "ph_cells": 0,
                    "sbs_cells": 0,
                    "status": "ERROR",
                }
            )

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    return summary_df


# Enable interactive backend
plt.ion()  # Turn on interactive mode


# Example usage:
"""
# Comprehensive well merge analysis
analyze_well_merge('1', 'B2', '/path/to/project/root')

# Quick analysis with default region  
quick_well_analysis('1', 'B2', '/path/to/project/root')

# Custom region analysis
analyze_well_merge('1', 'B2', '/path/to/project/root', region=(5000, 10000, 5000, 10000))

# Batch QC
wells_to_check = [(1, 'A01'), (1, 'A02'), (1, 'B01')]
summary = batch_qc_report('/path/to/analysis/merge', wells_to_check)
"""

def display_matched_and_unmatched_cells_for_site(root_fp, plate, well, selected_site=None, 
                                               distance_threshold=15.0, max_display_rows=1000):
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
        print(f"📁 Loading cell matching data for Plate {plate}, Well {well}...")
        merged_df = pd.read_parquet(merged_cells_path)
        phenotype_transformed = pd.read_parquet(phenotype_transformed_path)
        sbs_positions = pd.read_parquet(sbs_positions_path)
        
        print(f"✅ Loaded {len(merged_df)} total cell matches")
        print(f"✅ Loaded {len(phenotype_transformed)} transformed phenotype cells")
        print(f"✅ Loaded {len(sbs_positions)} SBS cells")
        
        # Get available sites from merged data
        available_sites = sorted(merged_df['site'].unique()) if 'site' in merged_df.columns else []
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
            print(f"   SBS coordinate range: i=[{sbs_i_min:.1f}, {sbs_i_max:.1f}], j=[{sbs_j_min:.1f}, {sbs_j_max:.1f}]")
            
            # For phenotype_transformed, filter by coordinates within SBS range
            site_phenotype = phenotype_transformed[
                (phenotype_transformed['i'] >= sbs_i_min) & 
                (phenotype_transformed['i'] <= sbs_i_max) &
                (phenotype_transformed['j'] >= sbs_j_min) & 
                (phenotype_transformed['j'] <= sbs_j_max)
            ].copy()
            print(f"   Phenotype cells within SBS coordinate range: {len(site_phenotype)}")
        else:
            print("❌ No SBS cells found for selected site")
            return None
        
        print(f"🔍 Site '{selected_site}' cell counts:")
        print(f"   Merged matches: {len(site_merged)}")
        print(f"   Matches within {distance_threshold}px: {len(filtered_merged)}")
        print(f"   Total phenotype cells (transformed): {len(site_phenotype)}")
        print(f"   Total SBS cells: {len(site_sbs)}")
        

        
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
        
        print(f"   Phenotype match rate: {match_rate_phenotype:.1%}")
        print(f"   SBS match rate: {match_rate_sbs:.1%}")
        
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
        
        # Return summary statistics
        summary_stats = {
            'site': selected_site,
            'total_phenotype_cells': total_phenotype,
            'total_sbs_cells': total_sbs,
            'matched_cells': len(filtered_merged),
            'unmatched_phenotype': len(unmatched_phenotype),
            'unmatched_sbs': len(unmatched_sbs),
            'phenotype_match_rate': match_rate_phenotype,
            'sbs_match_rate': match_rate_sbs,
            'mean_match_distance': distances.mean() if len(filtered_merged) > 0 else None,
            'matched_phenotype_stitched_ids': len(matched_phenotype_stitched_ids),
            'matched_sbs_stitched_ids': len(matched_sbs_stitched_ids)
        }
        
        return summary_stats
        
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
                                 distance_threshold=15.0, max_display_rows=1000):
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
        root_fp, plate, well, selected_site, distance_threshold, max_display_rows
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