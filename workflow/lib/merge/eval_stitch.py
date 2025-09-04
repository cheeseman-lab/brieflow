"""Quality Control and Evaluation for Stitched Well Outputs.

This module provides comprehensive quality control tools for evaluating stitched well images
and segmentation masks. It includes visualization capabilities, statistical analysis, and
interactive tools for examining stitching quality.

Run this in a Jupyter notebook for interactive QC capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")

def create_tile_arrangement_qc_plot(
    cell_positions_df: pd.DataFrame, 
    output_path: str, 
    data_type: str = "phenotype",
    well: str = None,
    plate: str = None
):
    """Create QC plot showing tile arrangement as a spatial heatmap with cell counts.
    
    Args:
        cell_positions_df: DataFrame with cell position data
        output_path: Path where to save the QC plot
        data_type: Type of data being plotted (default: "phenotype")
        well: Well identifier (optional)
        plate: Plate identifier (optional)
        
    Returns:
        Path to saved plot if successful, None if failed or no data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from pathlib import Path
    
    # Check if we have cell positions data
    if len(cell_positions_df) == 0:
        print("Skipping QC plot - no cell positions available")
        return None
    
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Use the correct column for preserved tile mapping
        tile_column = (
            "original_tile_id"
            if "original_tile_id" in cell_positions_df.columns
            else "tile"
        )
        
        if tile_column not in cell_positions_df.columns:
            ax.text(0.5, 0.5, f"No {tile_column} column found", 
                    ha="center", va="center", transform=ax.transAxes, fontsize=16)
            ax.set_title(f"{data_type.title()} Tile Arrangement QC - No Data")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None

        # Group by tile and get statistics
        tile_stats = cell_positions_df.groupby(tile_column).agg({
            'i': ['mean', 'count'],  # mean position and cell count
            'j': 'mean',
            'area': 'mean'
        }).reset_index()
        
        # Flatten column names
        tile_stats.columns = [tile_column, 'i_mean', 'cell_count', 'j_mean', 'area_mean']
        
        if len(tile_stats) == 0:
            ax.text(0.5, 0.5, "No tiles found", 
                    ha="center", va="center", transform=ax.transAxes, fontsize=16)
            ax.set_title(f"{data_type.title()} Tile Arrangement QC - No Data")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None
        
        # Create a grid layout based on tile positions
        # Convert stitched coordinates to grid positions
        i_positions = tile_stats['i_mean'].values
        j_positions = tile_stats['j_mean'].values
        
        # Normalize positions to create a regular grid
        # Find unique positions with some tolerance for floating point precision
        unique_i = np.unique(np.round(i_positions / 1000) * 1000)  # Round to nearest 1000
        unique_j = np.unique(np.round(j_positions / 1000) * 1000)
        
        # Create grid indices
        grid_rows = len(unique_i)
        grid_cols = len(unique_j)
        
        # Create the heatmap data
        heatmap_data = np.full((grid_rows, grid_cols), np.nan)
        tile_labels = np.full((grid_rows, grid_cols), "", dtype=object)
        
        # Map each tile to its grid position
        for _, row in tile_stats.iterrows():
            # Find closest grid position
            i_idx = np.argmin(np.abs(unique_i - row['i_mean']))
            j_idx = np.argmin(np.abs(unique_j - row['j_mean']))
            
            heatmap_data[i_idx, j_idx] = row['cell_count']
            tile_labels[i_idx, j_idx] = f"{int(row[tile_column])}\n({int(row['cell_count'])})"
        
        # Create custom colormap (similar to the reference image)
        colors = ['#2E1E66', '#3D4FB8', '#4FAADB', '#6FD4A8', '#A8E66F', '#EFEA4F']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Plot the heatmap
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='equal')
        
        # Add tile labels
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not np.isnan(heatmap_data[i, j]):
                    ax.text(j, i, tile_labels[i, j], 
                           ha="center", va="center", 
                           color="white", fontweight="bold", fontsize=8)
        
        # Customize the plot
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title with well and plate info if provided
        title_parts = [f"{data_type.title()} Tile Arrangement"]
        if plate and well:
            title_parts.append(f"Plate {plate}, Well {well}")
        ax.set_title(" - ".join(title_parts), fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cells per Tile', rotation=270, labelpad=20)
        
        # Add summary statistics as text
        total_cells = tile_stats['cell_count'].sum()
        mean_cells_per_tile = tile_stats['cell_count'].mean()
        total_tiles = len(tile_stats)
        
        stats_text = f"Total Cells: {total_cells:,}\nTiles: {total_tiles}\nMean/Tile: {mean_cells_per_tile:.0f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"QC plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"QC plot creation failed: {e}")
        return None