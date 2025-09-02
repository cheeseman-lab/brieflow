"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Plot merge example using regional sampling
    print("\n2. Plotting merge example with regional sampling...")
    plot_well_merge_example(alignment_data, threshold=threshold, sample_size=1000)
    
    return alignment_data


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
    def __init__(self, base_path, plate, well):
        """
        Initialize QC for a specific plate/well

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
        """Check which output files exist"""
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
        """Display phenotype and SBS overlays side by side"""
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
        """Analyze cell position data and create summary plots"""

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
        """
        Memory-efficient stitching quality check with interactive brightness/contrast controls
        
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
            """Apply brightness and contrast adjustments"""
            # Normalize to 0-1 range first
            img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            # Apply contrast (multiply) then brightness (add)
            adjusted = np.clip(contrast * img_norm + (brightness - 1.0), 0, 1)
            return adjusted
        
        def update_display(val=None):
            """Update all image displays with current slider values"""
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
        """
        Non-interactive version showing multiple brightness levels side by side
        Use this if interactive sliders don't work
        
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
            """Apply brightness adjustment"""
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
        """
        View a square region centered at specified coordinates with fixed brightness
        
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

    def view_mask_region(self, center_row, center_col, size=1000, modality="phenotype"):
        """
        View a square region from the stitched mask for the given modality.
        Now handles coordinates consistently with view_region.

        Parameters
        ----------
        center_row, center_col : int
            Center coordinates of region to view (in pixel space of stitched image/mask)
        size : int, default 1000
            Size of square region (size x size pixels)
        modality : str, default "phenotype"
            Which mask to view: "phenotype" or "sbs"
        """
        # Calculate region bounds (same as view_region)
        half_size = size // 2
        start_i = center_row - half_size
        end_i = center_row + half_size
        start_j = center_col - half_size
        end_j = center_col + half_size

        if modality == "phenotype":
            mask_path = self.phenotype_mask
            title = f"Phenotype Mask Region - Plate {self.plate}, Well {self.well}"
        elif modality == "sbs":
            mask_path = self.sbs_mask
            title = f"SBS Mask Region - Plate {self.plate}, Well {self.well}"
        else:
            raise ValueError("modality must be 'phenotype' or 'sbs'")

        if not mask_path.exists():
            print(f"❌ Mask file not found: {mask_path}")
            return

        print(f"Viewing {size}x{size} mask region centered at ({center_row}, {center_col})")
        print(f"Region bounds: [{start_i}:{end_i}, {start_j}:{end_j}]")

        # Load mask using memory mapping (consistent with image loading approach)
        mask = np.load(mask_path, mmap_mode="r")
        print(f"Mask shape: {mask.shape}")

        # Validate and clip region bounds to mask dimensions (EXACT same logic as image functions)
        start_i = max(0, min(start_i, mask.shape[0]))
        end_i = max(start_i, min(end_i, mask.shape[0]))
        start_j = max(0, min(start_j, mask.shape[1]))
        end_j = max(start_j, min(end_j, mask.shape[1]))

        # Extract region (now using numpy array conversion for consistency)
        region = np.array(mask[start_i:end_i, start_j:end_j])
        
        print(f"Extracted region shape: {region.shape}")
        print(f"Unique mask values: {np.unique(region)[:10]}...")  # Show first 10 unique values

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use appropriate colormap and display settings for masks
        im = ax.imshow(region, cmap="nipy_spectral", interpolation="nearest")
        ax.set_title(f"{title}\nCenter: ({center_row}, {center_col}), Actual region: {region.shape}\n"
                    f"Bounds: [{start_i}:{end_i}, {start_j}:{end_j}]")
        ax.axis("off")
        
        # Add colorbar to show cell ID scale
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cell ID', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Return the region for further analysis if needed
        return region
    

    def get_mask_info(self, modality="phenotype"):
        """
        Get basic information about a mask including dimensions and suggested viewing coordinates.
        
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


# Usage functions
def quick_qc(base_path, plate, well):
    """Quick QC check for a single well"""
    qc = StitchQC(base_path, plate, well)
    qc.view_overlays()
    return qc.analyze_cell_positions()


def batch_qc_report(base_path, plate_wells):
    """
    Generate QC reports for multiple wells

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



# Example usage:
"""
# Single well QC
qc = StitchQC('/path/to/analysis/merge', plate=1, well='A01')
qc.view_overlays()
ph_pos, sbs_pos = qc.analyze_cell_positions()

# Check specific region for stitching artifacts
qc.check_stitching_quality_efficient(sample_region=(5000, 6000, 8000, 9000))

# Alternative static version
qc.check_stitching_quality_static(sample_region=(5000, 6000, 8000, 9000))

# View specific region easily
qc.view_region(center_row=5500, center_col=8500, size=2000)

# Batch QC
wells_to_check = [(1, 'A01'), (1, 'A02'), (1, 'B01')]
summary = batch_qc_report('/path/to/analysis/merge', wells_to_check)
"""

