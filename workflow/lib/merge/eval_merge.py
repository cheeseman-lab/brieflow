"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import yaml

from lib.shared.eval import plot_plate_heatmap
from lib.merge.eval_alignment import plot_alignment_quality
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


def plot_well_alignment_quality(alignment_data, det_range, score, xlim=(0, 0.1), ylim=(0, 1), figsize=(10, 6)):
    """Plot alignment quality for well-based approach (single point).
    
    Adapts the tile-based plot_alignment_quality function to show a single well result.
    
    Args:
        alignment_data (dict): Output from load_well_alignment_outputs
        det_range (tuple): (min, max) range for acceptable determinant values
        score (float): Minimum acceptable score value
        xlim (tuple): X-axis limits
        ylim (tuple): Y-axis limits
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure object
        matplotlib.axes.Axes: The created axes object
    """
    if 'alignment_params' not in alignment_data:
        raise ValueError("No alignment parameters found in alignment_data")
    
    alignment_params = alignment_data['alignment_params'].iloc[0]
    
    # Create a DataFrame in the format expected by plot_alignment_quality
    df_align = pd.DataFrame({
        'determinant': [alignment_params['determinant']],
        'score': [alignment_params['score']],
        'tile': ['Phenotype Well'],  # Use descriptive labels
        'site': ['SBS Well']
    })
    
    # Use the existing function
    fig, ax = plot_alignment_quality(
        df_align, 
        det_range=det_range, 
        score=score, 
        xlim=xlim, 
        ylim=ylim, 
        figsize=figsize
    )
    
    # Update title to reflect well-based approach
    ax.set_title("Well-Level Alignment Quality Check\nScore vs Determinant")
    
    return fig, ax


def recreate_alignment_regions(alignment_data):
    """Recreate the same regional sampling used during well alignment.
    
    Args:
        alignment_data (dict): Output from load_well_alignment_outputs
        
    Returns:
        tuple: (phenotype_region, sbs_region, region_info)
    """
    # Get the region size from alignment summary
    region_size = alignment_data.get('alignment_summary', {}).get('alignment', {}).get('region_size', 7000)
    
    # Use the scaled phenotype positions (same as used during alignment)
    phenotype_scaled = alignment_data['phenotype_scaled']
    sbs_positions = alignment_data['sbs_positions']
    
    print(f"Recreating alignment regions with size: {region_size}")
    
    # Use the same regional sampling function that was used during alignment
    pheno_region, sbs_region, region_info = sample_region_for_alignment(
        phenotype_resized=phenotype_scaled,
        sbs_positions=sbs_positions,
        region_size=int(region_size),
        strategy="center"
    )
    
    return pheno_region, sbs_region, region_info


def plot_well_merge_example(alignment_data, threshold, sample_size=None, figsize=(30, 10)):
    """Visualize the well-level merge process using the same regions as alignment.
    
    Adapts plot_merge_example to work with well-level data and regional sampling.
    
    Args:
        alignment_data (dict): Output from load_well_alignment_outputs
        threshold (float): Distance threshold for matching points (use THRESHOLD from config)
        sample_size (int, optional): Number of cells to sample for visualization
        figsize (tuple): Figure size
    """
    # Recreate the same regions used during alignment
    df_ph_region, df_sbs_region, region_info = recreate_alignment_regions(alignment_data)
    
    if df_ph_region.empty or df_sbs_region.empty:
        print("❌ Failed to recreate alignment regions")
        return
    
    print(f"Using alignment regions: {len(df_ph_region)} phenotype, {len(df_sbs_region)} SBS cells")
    
    # Get transformation parameters
    alignment_params = alignment_data['alignment_params'].iloc[0]
    
    # Extract rotation and translation from saved parameters
    rotation_flat = alignment_params['rotation_matrix_flat']
    translation_list = alignment_params['translation_vector']
    
    # Reconstruct rotation matrix from flattened array
    rotation = np.array(rotation_flat).reshape(2, 2)
    translation = np.array(translation_list)
    
    print(f"Using transformation:")
    print(f"  Rotation: {rotation}")
    print(f"  Translation: {translation}")
    print(f"  Score: {alignment_params['score']:.3f}")
    print(f"  Determinant: {alignment_params['determinant']:.6f}")
    
    # Create a mock alignment_vec for compatibility with plot_merge_example
    alignment_vec = {
        'rotation': rotation,
        'translation': translation,
        'tile': 'Phenotype Region',  # Use descriptive names
        'site': 'SBS Region',
        'score': alignment_params['score'],
        'determinant': alignment_params['determinant']
    }
    
    # Sample data if requested
    if sample_size:
        if len(df_ph_region) > sample_size:
            df_ph_region = df_ph_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} phenotype cells for visualization")
        
        if len(df_sbs_region) > sample_size:
            df_sbs_region = df_sbs_region.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} SBS cells for visualization")
    
    # Use the existing plot_merge_example function
    plot_merge_example(
        df_ph=df_ph_region,
        df_sbs=df_sbs_region,
        alignment_vec=alignment_vec,
        threshold=threshold
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


# Example usage function
def run_well_alignment_qc(root_fp, plate, well, det_range, score, threshold):
    """Run complete QC visualization for a well alignment.
    
    Args:
        root_fp (str/Path): Root analysis directory
        plate (str): Plate identifier
        well (str): Well identifier
        det_range (tuple): Determinant range from config
        score (float): Score threshold from config
        threshold (float): Distance threshold from config
    """
    print(f"Running Well Alignment QC for Plate {plate}, Well {well}")
    print("-" * 60)
    
    # Load alignment data
    alignment_data = load_well_alignment_outputs(root_fp, plate, well)
    
    # Display summary
    display_well_alignment_summary(alignment_data)
    
    # Plot alignment quality (single point)
    print("\n1. Plotting alignment quality...")
    plot_well_alignment_quality(alignment_data, det_range=det_range, score=score)
    plt.show()
    
    # Plot merge example using regional sampling
    print("\n2. Plotting merge example with regional sampling...")
    plot_well_merge_example(alignment_data, threshold=threshold, sample_size=5000)
    
    return alignment_data
