"""Quality Control and Evaluation for Stitched Well Outputs.

This module provides comprehensive quality control tools for evaluating stitched well images
and segmentation masks. It includes visualization capabilities, statistical analysis, and
interactive tools for examining stitching quality.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import re
import matplotlib as mpl
from matplotlib.patches import Circle

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")


def plot_cell_positions_plate_scatter(
    parquet_files,
    plate: str = "6W",
    point_size: float = 0.1,
    alpha: float = 0.8,
    cmap: str = "tab20",
    title: str | None = None,
    tile_column: str | None = None,
    colorbar_label: str = "Original Tile ID",
    figsize: tuple[int, int] = (16, 11),
):
    """Create a plate-wide scatter plot of cell positions across all wells.

    This function visualizes cell positions from one or more parquet files
    in a plate layout (6-well, 24-well, or 96-well). Each subplot corresponds
    to a well, and cells are displayed with consistent aesthetics similar to
    a heatmap.

    Parameters
    ----------
    parquet_files : str | Path | list[str | Path]
        One or multiple parquet file paths containing cell position data.
    plate : {"6W", "24W", "96W"}, default="6W"
        Plate format to determine well grid layout.
    point_size : float, default=0.1
        Size of each cell scatter point.
    alpha : float, default=0.8
        Transparency of scatter points.
    cmap : str, default="tab20"
        Colormap used to distinguish tile IDs (or categories from `tile_column`).
    title : str, optional
        Global title displayed above the entire plate layout.
    tile_column : str, optional
        Column name in the parquet file to use for coloring points.
        If None, defaults to "tile" or similar column in the data.
    colorbar_label : str, default="Original Tile ID"
        Label for the colorbar on the right side of the figure.
    figsize : tuple of int, default=(16, 11)
        Overall figure size in inches (width, height).

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plate scatter plot.
    axes : ndarray of matplotlib.axes.Axes
        The grid of subplot Axes, indexed by well position.

    Notes:
    -----
    - Subplots are arranged in a plate-like grid.
    - Each well plot has no axis lines or ticks, and points are masked
      to a circular outline.
    - A shared colorbar is placed far to the right for clarity.
    - Designed for visual QC of cell positions across plates.
    """
    # ----------------- Load & concat -----------------
    dfs = []
    for fp in parquet_files:
        p = Path(fp)
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
        df = pd.read_parquet(p)
        # Extract well name (A1..H12) from filename if not present
        if "well" not in df.columns:
            m = re.search(r"([A-H]\d{1,2})", p.stem)
            if not m:
                raise ValueError(f"Could not infer 'well' from filename: {p.name}")
            df["well"] = m.group(1)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid parquet files provided.")
    cell_positions_df = pd.concat(dfs, ignore_index=True)

    # tile column
    if tile_column is None:
        if "original_tile_id" in cell_positions_df.columns:
            tile_column = "original_tile_id"
        elif "tile" in cell_positions_df.columns:
            tile_column = "tile"
        else:
            tile_column = None  # allowed: plot in single color

    # ----------------- Plate layout -----------------
    plate_dims = {"6W": (2, 3), "24W": (4, 6), "96W": (8, 12)}
    if plate not in plate_dims:
        raise ValueError(f"Unsupported plate: {plate}")
    nrows, ncols = plate_dims[plate]

    # map wells to grid positions
    well_pos = {}
    if plate == "6W":
        order = ["A1", "A2", "A3", "B1", "B2", "B3"]
        for i, w in enumerate(order):
            well_pos[w] = (i // 3, i % 3)
    else:
        # generic A.. rows, 1.. cols
        row_letters = nrows
        col_nums = ncols
        for r in range(row_letters):
            for c in range(col_nums):
                well_pos[f"{chr(65 + r)}{c + 1}"] = (r, c)

    # ----------------- Style to match heatmap -----------------
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # matches matplotlib default look in your heatmap
            "font.size": 28,  # Increased base font size
            "axes.linewidth": 0.0,  # no axis lines
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "axes.titlesize": 32,  # Explicit title size
        }
    )

    # Create figure with more generous spacing
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        gridspec_kw={
            "right": 0.83,  # More generous right margin for colorbar
            "hspace": 0.4,  # Increased vertical spacing between subplots
            "wspace": 0.4,  # Increased horizontal spacing between subplots
        },
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # ----------------- Plot per well -----------------
    unique_wells = sorted(
        cell_positions_df["well"].unique(), key=lambda x: (x[0], int(x[1:]))
    )

    last_scatter = None
    for well in well_pos:
        r, c = well_pos[well]
        ax = axes[r, c]

        # clear axis cosmetics to match heatmap
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        wdf = cell_positions_df[cell_positions_df["well"] == well]
        if wdf.empty:
            ax.set_title(f"Well {well}", fontsize=24)  # Explicit font size
            ax.set_frame_on(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            continue

        # determine circle center/limits from data extents
        jmin, jmax = wdf["j"].min(), wdf["j"].max()
        imin, imax = wdf["i"].min(), wdf["i"].max()
        cx, cy = (jmin + jmax) / 2.0, (imin + imax) / 2.0
        r_pix = 0.5 * min(jmax - jmin, imax - imin) * 0.98  # small padding

        # set square limits centered on the well
        ax.set_xlim(cx - r_pix, cx + r_pix)
        ax.set_ylim(cy - r_pix, cy + r_pix)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        # circular clip so points outside the well are hidden - NO EDGE COLOR
        circle = Circle(
            (cx, cy), r_pix, facecolor="none", edgecolor="none", linewidth=0
        )
        ax.add_patch(circle)

        # scatter
        if tile_column and tile_column in wdf.columns:
            last_scatter = ax.scatter(
                wdf["j"],
                wdf["i"],
                c=wdf[tile_column],
                cmap=cmap,
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            # clip the scatter to the circle
            last_scatter.set_clip_path(circle)
        else:
            last_scatter = ax.scatter(
                wdf["j"],
                wdf["i"],
                s=point_size,
                alpha=alpha,
                color="k",
                linewidths=0,
            )
            last_scatter.set_clip_path(circle)

        # well title with explicit font size
        ax.set_title(f"Well {well}", fontsize=24)

    # hide any axes that aren't part of this plate
    for r in range(nrows):
        for c in range(ncols):
            # if no matching well in mapping, hide
            if all(well_pos.get(w) != (r, c) for w in well_pos):
                axes[r, c].set_visible(False)

    # global title (match your heatmap → usually none)
    if title:
        fig.suptitle(title, y=0.98, fontsize=24)

    # ----------------- Colorbar far right -----------------
    if last_scatter is not None:
        cax = fig.add_axes([0.86, 0.15, 0.025, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label(colorbar_label, rotation=270, labelpad=25, fontsize=20)
        cbar.ax.tick_params(labelsize=18)  # Colorbar tick label size

    # Use less restrictive layout - let matplotlib handle most of the spacing
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.83, top=0.93)

    return fig, axes


def create_tile_arrangement_qc_plot(
    cell_positions_df: pd.DataFrame,
    output_path: str,
    data_type: str = "phenotype",
    well: str = None,
    plate: str = None,
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
            ax.text(
                0.5,
                0.5,
                f"No {tile_column} column found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title(f"{data_type.title()} Tile Arrangement QC - No Data")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None

        # Group by tile and get statistics
        tile_stats = (
            cell_positions_df.groupby(tile_column)
            .agg(
                {
                    "i": ["mean", "count"],  # mean position and cell count
                    "j": "mean",
                    "area": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        tile_stats.columns = [
            tile_column,
            "i_mean",
            "cell_count",
            "j_mean",
            "area_mean",
        ]

        if len(tile_stats) == 0:
            ax.text(
                0.5,
                0.5,
                "No tiles found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title(f"{data_type.title()} Tile Arrangement QC - No Data")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None

        # Create a grid layout based on tile positions
        # Convert stitched coordinates to grid positions
        i_positions = tile_stats["i_mean"].values
        j_positions = tile_stats["j_mean"].values

        # Normalize positions to create a regular grid
        # Find unique positions with some tolerance for floating point precision
        unique_i = np.unique(
            np.round(i_positions / 1000) * 1000
        )  # Round to nearest 1000
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
            i_idx = np.argmin(np.abs(unique_i - row["i_mean"]))
            j_idx = np.argmin(np.abs(unique_j - row["j_mean"]))

            heatmap_data[i_idx, j_idx] = row["cell_count"]
            tile_labels[i_idx, j_idx] = (
                f"{int(row[tile_column])}\n({int(row['cell_count'])})"
            )

        # Create custom colormap (similar to the reference image)
        colors = ["#2E1E66", "#3D4FB8", "#4FAADB", "#6FD4A8", "#A8E66F", "#EFEA4F"]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        # Plot the heatmap
        im = ax.imshow(heatmap_data, cmap=cmap, aspect="equal")

        # Add tile labels
        for i in range(grid_rows):
            for j in range(grid_cols):
                if not np.isnan(heatmap_data[i, j]):
                    ax.text(
                        j,
                        i,
                        tile_labels[i, j],
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=8,
                    )

        # Customize the plot
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title with well and plate info if provided
        title_parts = [f"{data_type.title()} Tile Arrangement"]
        if plate and well:
            title_parts.append(f"Plate {plate}, Well {well}")
        ax.set_title(" - ".join(title_parts), fontsize=14, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cells per Tile", rotation=270, labelpad=20)

        # Add summary statistics as text
        total_cells = tile_stats["cell_count"].sum()
        mean_cells_per_tile = tile_stats["cell_count"].mean()
        total_tiles = len(tile_stats)

        stats_text = f"Total Cells: {total_cells:,}\nTiles: {total_tiles}\nMean/Tile: {mean_cells_per_tile:.0f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"QC plot saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"QC plot creation failed: {e}")
        return None

def create_tile_arrangement_qc_plot(
    cell_positions_df: pd.DataFrame,
    output_path: str,
    data_type: str = "phenotype",
    well: str = None,
    plate: str = None,
):
    """Create QC plot showing tile arrangement and cell distribution.

    This function generates a comprehensive quality control plot that visualizes
    tile arrangement, cell distributions, and spatial relationships using scatter plots
    similar to plot_cell_positions_plate_scatter.

    Parameters
    ----------
    cell_positions_df : pd.DataFrame
        Cell positions dataframe with tile information
    output_path : str
        Path where to save the QC plot
    data_type : str, default "phenotype"
        Type of data for plot title
    well : str, optional
        Well identifier for title
    plate : str, optional
        Plate identifier for title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Check if we have cell positions data
    if len(cell_positions_df) == 0:
        print("Skipping QC plot - no cell positions available")
        # Create a minimal placeholder plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        title_parts = [f"{data_type.title()} Tile Arrangement QC"]
        if plate and well:
            title_parts.append(f"Plate {plate}, Well {well}")
        ax.text(
            0.5,
            0.5,
            f"No cells found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(" - ".join(title_parts))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None

    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Create title
        title_parts = [f"{data_type.title()} Tile Arrangement QC"]
        if plate and well:
            title_parts.append(f"Plate {plate}, Well {well}")
        fig.suptitle(" - ".join(title_parts), fontsize=16)

        # Use the correct column for tile mapping
        tile_column = (
            "original_tile_id"
            if "original_tile_id" in cell_positions_df.columns
            else "tile"
        )

        # Plot 1: Cell positions colored by tile
        ax1 = axes[0, 0]
        if tile_column in cell_positions_df.columns:
            scatter = ax1.scatter(
                cell_positions_df["j"],
                cell_positions_df["i"],
                c=cell_positions_df[tile_column],
                cmap="tab20",
                s=0.1,
                alpha=0.7,
                linewidths=0,
            )
            plt.colorbar(scatter, ax=ax1, label="Original Tile ID")
        else:
            ax1.scatter(
                cell_positions_df["j"],
                cell_positions_df["i"],
                s=0.1,
                alpha=0.7,
                color="k",
                linewidths=0,
            )
        ax1.set_title("Cell Positions by Original Tile")
        ax1.set_xlabel("X Position (pixels)")
        ax1.set_ylabel("Y Position (pixels)")
        ax1.invert_yaxis()
        ax1.set_aspect("equal", adjustable="box")

        # Plot 2: Stage coordinates with tile numbers (if available)
        ax2 = axes[0, 1]
        if (
            "stage_x" in cell_positions_df.columns
            and "stage_y" in cell_positions_df.columns
        ):
            # Get unique tile positions
            tile_info = (
                cell_positions_df.groupby(tile_column)
                .agg({"stage_x": "first", "stage_y": "first"})
                .reset_index()
            )

            # Filter out NaN values
            tile_info = tile_info.dropna(subset=["stage_x", "stage_y"])

            if len(tile_info) > 0:
                ax2.scatter(tile_info["stage_x"], tile_info["stage_y"], s=50)
                for _, row in tile_info.iterrows():
                    ax2.annotate(
                        f"{int(row[tile_column])}",
                        (row["stage_x"], row["stage_y"]),
                        fontsize=8,
                        ha="center",
                    )
                ax2.set_xlabel("Stage X (μm)")
                ax2.set_ylabel("Stage Y (μm)")
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No stage coordinates available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No stage coordinates available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        ax2.set_title("Tile Arrangement (Stage Coordinates)")

        # Plot 3: Cells per tile histogram
        ax3 = axes[1, 0]
        if tile_column in cell_positions_df.columns:
            tile_counts = cell_positions_df[tile_column].value_counts()
            if len(tile_counts) > 0:
                ax3.hist(tile_counts.values, bins=min(50, len(tile_counts)), alpha=0.7)
                ax3.axvline(
                    tile_counts.mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {tile_counts.mean():.1f}",
                )
                ax3.legend()
                ax3.set_xlabel("Cells per Tile")
                ax3.set_ylabel("Number of Tiles")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No tile data available",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "No tile column found",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
        ax3.set_title("Distribution of Cells per Original Tile")

        # Plot 4: Relative positions within original tiles (if available)
        ax4 = axes[1, 1]
        if (
            "tile_i" in cell_positions_df.columns
            and "tile_j" in cell_positions_df.columns
        ):
            # Filter out invalid relative positions
            valid_positions = cell_positions_df[
                (cell_positions_df["tile_i"] >= 0) & (cell_positions_df["tile_j"] >= 0)
            ]

            if len(valid_positions) > 0 and tile_column in valid_positions.columns:
                ax4.scatter(
                    valid_positions["tile_j"],
                    valid_positions["tile_i"],
                    c=valid_positions[tile_column],
                    cmap="tab20",
                    s=0.1,
                    alpha=0.7,
                    linewidths=0,
                )
                ax4.set_xlabel("Tile-relative X")
                ax4.set_ylabel("Tile-relative Y")
                ax4.invert_yaxis()
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No valid relative positions",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
        else:
            ax4.text(
                0.5,
                0.5,
                "No relative position data available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
        ax4.set_title("Relative Positions within Original Tiles")

        # Add summary statistics
        total_cells = len(cell_positions_df)
        if tile_column in cell_positions_df.columns:
            unique_tiles = cell_positions_df[tile_column].nunique()
            mean_cells_per_tile = total_cells / unique_tiles if unique_tiles > 0 else 0
        else:
            unique_tiles = 0
            mean_cells_per_tile = 0

        stats_text = f"Total Cells: {total_cells:,}\nTiles: {unique_tiles}\nMean/Tile: {mean_cells_per_tile:.0f}"

        # Add stats text to the figure
        fig.text(
            0.02,
            0.02,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"QC plot saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"QC plot creation failed: {e}")
        # Create error plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        title_parts = [f"{data_type.title()} Tile Arrangement QC"]
        if plate and well:
            title_parts.append(f"Plate {plate}, Well {well}")
        ax.text(
            0.5,
            0.5,
            f"Error creating plot: {str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(" - ".join(title_parts))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None

