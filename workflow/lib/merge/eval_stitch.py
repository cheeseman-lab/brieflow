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

warnings.filterwarnings("ignore")

# Set up plotting
plt.style.use("default")


import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle

def plot_cell_positions_plate_scatter(
    parquet_files,
    plate="6W",
    point_size=0.1,
    alpha=0.8,
    cmap="tab20",
    title=None,                  # keep None to match your heatmap (no suptitle)
    tile_column=None,            # auto-detect if None
    colorbar_label="Original Tile ID",
    figsize=(16, 11),            # Increased figure size to give more room
):
    """
    Step 1: Concatenate parquets
    Step 2: Make a well-layout grid (6W/24W/96W)
    Step 3: Scatter per well with heatmap-like aesthetics:
            - same font size feel
            - no axis lines/ticks
            - circular mask
            - colorbar far right
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
            m = re.search(r'([A-H]\d{1,2})', p.stem)
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
        order = ["A1","A2","A3","B1","B2","B3"]
        for i, w in enumerate(order):
            well_pos[w] = (i // 3, i % 3)
    else:
        # generic A.. rows, 1.. cols
        row_letters = nrows
        col_nums = ncols
        for r in range(row_letters):
            for c in range(col_nums):
                well_pos[f"{chr(65+r)}{c+1}"] = (r, c)

    # ----------------- Style to match heatmap -----------------
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",  # matches matplotlib default look in your heatmap
        "font.size": 28,                # Increased base font size
        "axes.linewidth": 0.0,          # no axis lines
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.titlesize": 32,           # Explicit title size
    })

    # Create figure with more generous spacing
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        gridspec_kw={
            "right": 0.83,      # More generous right margin for colorbar
            "hspace": 0.4,      # Increased vertical spacing between subplots
            "wspace": 0.4       # Increased horizontal spacing between subplots
        }
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    # ----------------- Plot per well -----------------
    unique_wells = sorted(cell_positions_df["well"].unique(), key=lambda x: (x[0], int(x[1:])))

    last_scatter = None
    for well in well_pos:
        r, c = well_pos[well]
        ax = axes[r, c]

        # clear axis cosmetics to match heatmap
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        wdf = cell_positions_df[cell_positions_df["well"] == well]
        if wdf.empty:
            ax.set_title(f"Well {well}", fontsize=24)  # Explicit font size
            ax.set_frame_on(False)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
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
        circle = Circle((cx, cy), r_pix, facecolor="none", edgecolor="none", linewidth=0)
        ax.add_patch(circle)

        # scatter
        if tile_column and tile_column in wdf.columns:
            last_scatter = ax.scatter(
                wdf["j"], wdf["i"],
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
                wdf["j"], wdf["i"],
                s=point_size, alpha=alpha, color="k", linewidths=0,
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
