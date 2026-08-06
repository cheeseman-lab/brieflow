"""Shared utilties for configuring Brieflow process parameters.

This includes:
- Functions for viewing steps of merge process such as determining tiles to merge and seeing an example merge.
"""

import re
import math
from pathlib import Path

import pandas as pd
from microfilm.microplot import Micropanel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.morphology
import matplotlib.colors as mcolors
from skimage.measure import regionprops


from lib.merge.fast_merge import build_linear_model, refine_local_warp, match_cells


def plot_combined_tile_grid(
    ph_metadata,
    sbs_metadata,
    ph_image_dims=(2960, 2960),
    sbs_image_dims=(1480, 1480),
    figsize=None,
):
    """Plots a combined grid of X-Y positions for PH and SBS datasets as rectangles.

    Tile sizes are calculated dynamically from pixel_size metadata if available,
    otherwise estimated from coordinate spacing. Labels are centered inside tiles
    with auto-scaled font sizes.

    Args:
        ph_metadata (pd.DataFrame): DataFrame containing PH metadata with columns:
            'x_pos', 'y_pos', 'tile', and optionally 'pixel_size_x'.
        sbs_metadata (pd.DataFrame): DataFrame containing SBS metadata with columns:
            'x_pos', 'y_pos', 'tile', and optionally 'pixel_size_x'.
        ph_image_dims (tuple, optional): Phenotype image dimensions (height, width)
            in pixels. Used with pixel_size to calculate tile size. Defaults to (2960, 2960).
        sbs_image_dims (tuple, optional): SBS image dimensions (height, width)
            in pixels. Used with pixel_size to calculate tile size. Defaults to (1480, 1480).
        figsize (tuple, optional): Figure size. If None, auto-calculated based on
            data extent. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Calculate tile sizes
    if "pixel_size_x" in ph_metadata.columns:
        ph_tile_size = ph_image_dims[0] * ph_metadata["pixel_size_x"].iloc[0]
    else:
        ph_tile_size = _estimate_tile_size_from_coords(ph_metadata)

    if "pixel_size_x" in sbs_metadata.columns:
        sbs_tile_size = sbs_image_dims[0] * sbs_metadata["pixel_size_x"].iloc[0]
    else:
        sbs_tile_size = _estimate_tile_size_from_coords(sbs_metadata)

    # Auto-calculate figure size based on data extent
    all_x = pd.concat([ph_metadata["x_pos"], sbs_metadata["x_pos"]])
    all_y = pd.concat([ph_metadata["y_pos"], sbs_metadata["y_pos"]])
    x_range = all_x.max() - all_x.min() + max(ph_tile_size, sbs_tile_size)
    y_range = all_y.max() - all_y.min() + max(ph_tile_size, sbs_tile_size)
    aspect = x_range / y_range if y_range > 0 else 1

    if figsize is None:
        figsize = (min(24, 14 * aspect), 14)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Draw PH tiles as rectangles with labels
    for _, row in ph_metadata.iterrows():
        rect = mpatches.Rectangle(
            (row["x_pos"], row["y_pos"]),
            ph_tile_size,
            ph_tile_size,
            linewidth=0.5,
            edgecolor="black",
            facecolor="white",
            alpha=0.7,
        )
        ax.add_patch(rect)
        # Label centered in tile
        ax.text(
            row["x_pos"] + ph_tile_size / 2,
            row["y_pos"] + ph_tile_size / 2,
            str(int(row["tile"])),
            ha="center",
            va="center",
            fontsize=_auto_fontsize(ph_tile_size, x_range),
            color="black",
        )

    # Draw SBS tiles as rectangles with labels
    for _, row in sbs_metadata.iterrows():
        rect = mpatches.Rectangle(
            (row["x_pos"], row["y_pos"]),
            sbs_tile_size,
            sbs_tile_size,
            linewidth=0.5,
            edgecolor="darkred",
            facecolor="red",
            alpha=0.4,
        )
        ax.add_patch(rect)
        ax.text(
            row["x_pos"] + sbs_tile_size / 2,
            row["y_pos"] + sbs_tile_size / 2,
            str(int(row["tile"])),
            ha="center",
            va="center",
            fontsize=_auto_fontsize(sbs_tile_size, x_range),
            color="darkred",
            fontweight="bold",
        )

    # Create legend patches
    ph_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", alpha=0.7, label="PH"
    )
    sbs_patch = mpatches.Patch(
        facecolor="red", edgecolor="darkred", alpha=0.4, label="SBS"
    )
    ax.legend(handles=[ph_patch, sbs_patch], fontsize=12, loc="upper right")

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("X Position (µm)", fontsize=14)
    ax.set_ylabel("Y Position (µm)", fontsize=14)
    ax.set_title("Combined Tile Grid - PH (white) & SBS (red)", fontsize=16)

    plt.tight_layout()
    return fig


def _auto_fontsize(tile_size, plot_range, min_size=4, max_size=10):
    """Calculate font size based on tile size relative to plot range.

    Args:
        tile_size (float): Size of the tile in plot coordinates.
        plot_range (float): Total range of the plot (max - min).
        min_size (int, optional): Minimum font size. Defaults to 4.
        max_size (int, optional): Maximum font size. Defaults to 10.

    Returns:
        float: Calculated font size.
    """
    fraction = tile_size / plot_range
    size = min_size + (max_size - min_size) * min(1, fraction * 20)
    return max(min_size, min(max_size, size))


def _estimate_tile_size_from_coords(metadata):
    """Estimate tile size from coordinate spacing.

    Args:
        metadata (pd.DataFrame): DataFrame with 'x_pos' and 'y_pos' columns.

    Returns:
        float: Estimated tile size based on median spacing between adjacent tiles.
    """
    sorted_x = metadata["x_pos"].sort_values().diff().dropna()
    sorted_y = metadata["y_pos"].sort_values().diff().dropna()
    # Use median of non-zero diffs as spacing
    x_spacing = sorted_x[sorted_x > 0].median()
    y_spacing = sorted_y[sorted_y > 0].median()
    return min(x_spacing, y_spacing) if pd.notna(x_spacing) else 1000


def plot_merge_example(
    df_ph, df_sbs, alignment_vec, threshold=2, local_refinement=None, warp_kwargs=None
):
    """Visualizes the merge process for a single tile-site pair.

    Args:
        df_ph (pandas.DataFrame): Phenotype data with 'i', 'j' columns.
        df_sbs (pandas.DataFrame): SBS data with 'i', 'j' columns.
        alignment_vec (dict): Contains 'rotation' and 'translation' for alignment.
        threshold (float, optional): Distance threshold for matching points. Defaults to 2.
        local_refinement (str | bool | None, optional): Warp model to apply to the affine
            prediction before matching, matching the pipeline (`refine_local_warp`). Defaults None.
        warp_kwargs (dict | None, optional): Keyword args forwarded to `refine_local_warp`. Defaults None.
    """
    # Create the figure — three panels sharing one matched/unmatched coloring
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    # Filter for the specific tile and site
    df_ph_filtered = df_ph[df_ph["tile"] == alignment_vec["tile"]]
    df_sbs_filtered = df_sbs[df_sbs["tile"] == alignment_vec["site"]]

    X = df_ph_filtered[["i", "j"]].values
    Y = df_sbs_filtered[["i", "j"]].values

    # Predict phenotype coordinates into SBS space, optionally warped (mirrors the pipeline)
    model = build_linear_model(alignment_vec["rotation"], alignment_vec["translation"])
    Y_pred = model.predict(X)
    if local_refinement:
        wk = dict(warp_kwargs or {})
        if isinstance(local_refinement, str):
            wk.setdefault("model", local_refinement)
        Y_pred = refine_local_warp(X, Y, Y_pred, threshold, **wk)

    # Mutual nearest-neighbor match — the same 1:1 rule merge_sbs_phenotype uses
    sbs_ix, ph_ix, match_distances = match_cells(Y, Y_pred, threshold)
    n_ph, n_sbs, n_matched = len(X), len(Y), len(ph_ix)
    matched_ph_mask = np.zeros(n_ph, dtype=bool)
    matched_ph_mask[ph_ix] = True
    n_unmatched = int((~matched_ph_mask).sum())
    frac_ph = n_matched / n_ph if n_ph else 0.0
    median_residual = float(np.median(match_distances)) if n_matched else float("nan")
    doubles = n_matched - len(np.unique(ph_ix))

    # Header carries the merge stats so each preview is self-describing
    fig.suptitle(
        f"PH tile {alignment_vec['tile']} ↔ SBS site {alignment_vec['site']}   |   "
        f"{n_matched} matched   |   {frac_ph * 100:.0f}% phenotype   |   "
        f"{median_residual:.2f} px median residual   |   {doubles} doubles",
        fontsize=16,
    )

    # Panel 1: matched/unmatched phenotype in the real SBS pixel frame, with residual segments
    ax1.scatter(
        Y[:, 0], Y[:, 1], c="lightgray", s=8, alpha=0.3, label=f"SBS field ({n_sbs})"
    )
    for k in range(n_matched):
        p, q = Y_pred[ph_ix[k]], Y[sbs_ix[k]]
        ax1.plot([p[0], q[0]], [p[1], q[1]], "k-", alpha=0.3, linewidth=0.5)
    ax1.scatter(
        Y_pred[matched_ph_mask, 0],
        Y_pred[matched_ph_mask, 1],
        c="#2f6fb0",
        s=14,
        alpha=0.7,
        label=f"matched phenotype ({n_matched})",
    )
    ax1.scatter(
        Y_pred[~matched_ph_mask, 0],
        Y_pred[~matched_ph_mask, 1],
        marker="*",
        c="#e8b93a",
        s=45,
        alpha=0.9,
        label=f"unmatched phenotype ({n_unmatched})",
    )
    ax1.set_aspect("equal")
    ax1.set_title("Aligned overlay (SBS pixel space)")
    ax1.legend(loc="upper right", fontsize=9)

    # Normalize phenotype coordinates into the SBS field for the scaled panels
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = (X_norm * (Y_pred.max(axis=0) - Y_pred.min(axis=0))) + Y_pred.min(axis=0)

    # Panel 2: normalized overlap of phenotype on the SBS field
    ax2.scatter(
        Y[:, 0], Y[:, 1], c="lightgray", s=12, alpha=0.15, label=f"SBS field ({n_sbs})"
    )
    ax2.scatter(
        Y_pred[matched_ph_mask, 0],
        Y_pred[matched_ph_mask, 1],
        c="#c0392b",
        s=14,
        alpha=0.3,
        label=f"aligned SBS matches ({n_matched})",
    )
    ax2.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c="#2f6fb0",
        s=14,
        alpha=0.25,
        label=f"phenotype ({n_ph})",
    )
    ax2.set_title("Normalized phenotype relative to SBS")
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: matched vs unmatched phenotype in the normalized frame (no per-cell labels)
    ax3.scatter(
        Y[:, 0], Y[:, 1], c="lightgray", s=12, alpha=0.15, label=f"SBS field ({n_sbs})"
    )
    ax3.scatter(
        X_scaled[matched_ph_mask, 0],
        X_scaled[matched_ph_mask, 1],
        c="#2f6fb0",
        s=14,
        alpha=0.35,
        label=f"matched phenotype ({n_matched})",
    )
    ax3.scatter(
        X_scaled[~matched_ph_mask, 0],
        X_scaled[~matched_ph_mask, 1],
        marker="*",
        c="#e8b93a",
        s=60,
        alpha=0.9,
        label=f"unmatched phenotype ({n_unmatched})",
    )
    ax3.set_title("Matched vs unmatched phenotype")
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def preview_mask_transformations(
    metadata,
    root_fp=None,
    data_type="phenotype",
    mask_type="nuclei",
    num_tiles=15,
    flipud=False,
    fliplr=False,
    rot90=0,
    figsize=(20, 10),
):
    """Preview mask transformations (flipud, fliplr, rot90) on the first N tiles.

    Arranged according to coordinate-based stitching estimates.
    """
    if root_fp is None:
        root_fp = Path("/lab/ops_analysis/lourido/nebo-analysis/analysis/analysis_root")
    else:
        root_fp = Path(root_fp)

    # Select only the first N tiles to preview
    first_tiles = metadata.head(num_tiles).copy()
    well = first_tiles["well"].iloc[0]

    print(
        f"Testing transformations on first {len(first_tiles)} {data_type} tiles ({mask_type} masks)"
    )
    print(f"Transformation: flipud={flipud}, fliplr={fliplr}, rot90={rot90}")

    # --- determine pixel scaling from metadata or fallback ---
    if data_type == "sbs":
        tile_size = (1200, 1200)
        fov_um = 1560.0
    else:
        tile_size = (2400, 2400)
        fov_um = 260.0

    # Use pixel size from metadata if available
    if "pixel_size_x" in first_tiles.columns and "pixel_size_y" in first_tiles.columns:
        pixel_size_um = first_tiles["pixel_size_x"].iloc[0]
        pixels_per_micron = 1.0 / pixel_size_um
    else:
        pixels_per_micron = tile_size[0] / fov_um

    # --- compute pixel positions for preview tiles ---
    coords_um = first_tiles[["x_pos", "y_pos"]].values
    x_min, y_min = coords_um.min(axis=0)
    translations = {}
    for idx, row in first_tiles.iterrows():
        x_pos, y_pos = row["x_pos"], row["y_pos"]
        pixel_x = int((x_pos - x_min) * pixels_per_micron)
        pixel_y = int((y_pos - y_min) * pixels_per_micron)
        translations[f"{row['well']}/{row['tile']}"] = [pixel_y, pixel_x]

    # Store centroids instead of full tiles
    original_centroids = []
    transformed_centroids = []
    files_found = 0

    for _, row in first_tiles.iterrows():
        try:
            filename = (
                f"P-{row['plate']}_W-{row['well']}_T-{row['tile']}__{mask_type}.tiff"
            )
            tile_path = root_fp / data_type / "images" / filename

            if tile_path.exists():
                try:
                    import tifffile

                    tile_data = tifffile.imread(str(tile_path))
                except ImportError:
                    from PIL import Image

                    tile_data = np.array(Image.open(str(tile_path)))
                files_found += 1

                if tile_data.ndim > 2:
                    if tile_data.shape[0] < 10:  # channels-first
                        tile_data = np.max(tile_data, axis=0)
                    else:
                        tile_data = (
                            tile_data[..., 0]
                            if tile_data.shape[-1] < 10
                            else tile_data[0]
                        )
            else:
                tile_data = np.zeros(tile_size, dtype=np.uint16)
                tile_data[100:150, 100:150] = row["tile"] % 255

            # Get tile offset
            y_offset, x_offset = translations[f"{row['well']}/{row['tile']}"]

            # Extract centroids from original tile
            props = regionprops(tile_data.astype(int))
            if len(props) > 0:
                centroids = np.array(
                    [
                        [p.centroid[0] + y_offset, p.centroid[1] + x_offset]
                        for p in props
                    ]
                )
                original_centroids.append(centroids)

            # Apply transformations
            transformed_tile = tile_data.copy()
            if rot90 > 0:
                transformed_tile = np.rot90(transformed_tile, k=rot90)
            if flipud:
                transformed_tile = np.flipud(transformed_tile)
            if fliplr:
                transformed_tile = np.fliplr(transformed_tile)

            # Extract centroids from transformed tile
            props_trans = regionprops(transformed_tile.astype(int))
            if len(props_trans) > 0:
                centroids_trans = np.array(
                    [
                        [p.centroid[0] + y_offset, p.centroid[1] + x_offset]
                        for p in props_trans
                    ]
                )
                transformed_centroids.append(centroids_trans)

        except Exception as e:
            print(f"Error loading tile {row['tile']}: {e}")

    print(f"Successfully loaded {files_found}/{len(first_tiles)} mask files")

    # Combine all centroids
    all_original = (
        np.vstack(original_centroids)
        if original_centroids
        else np.array([]).reshape(0, 2)
    )
    all_transformed = (
        np.vstack(transformed_centroids)
        if transformed_centroids
        else np.array([]).reshape(0, 2)
    )

    # --- compute overall axis limits ---
    if len(all_original) > 0:
        y_min_plot = min(all_original[:, 0].min(), all_transformed[:, 0].min())
        y_max_plot = max(all_original[:, 0].max(), all_transformed[:, 0].max())
        x_min_plot = min(all_original[:, 1].min(), all_transformed[:, 1].min())
        x_max_plot = max(all_original[:, 1].max(), all_transformed[:, 1].max())
    else:
        y_min_plot, y_max_plot = 0, tile_size[0]
        x_min_plot, x_max_plot = 0, tile_size[1]

    # --- plot side by side ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, centroids, title in zip(
        axes,
        [all_original, all_transformed],
        [
            "Original Tiles",
            f"Transformed (flipud={flipud}, fliplr={fliplr}, rot90={rot90})",
        ],
    ):
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_aspect("equal")

        if len(centroids) > 0:
            ax.scatter(
                centroids[:, 1],
                centroids[:, 0],
                s=10,
                alpha=0.6,
                c="red",
                edgecolors="none",
            )

        # Set axis limits
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.invert_yaxis()  # Match image coordinate convention
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(
        f"Total objects: {len(all_original)} (original), {len(all_transformed)} (transformed)"
    )

    return {"flipud": flipud, "fliplr": fliplr, "rot90": rot90}


def align_metadata(
    df1,
    df2,
    x_col="x_pos",
    y_col="y_pos",
    reference_df=2,
    flip_x=False,
    flip_y=False,
    rotate_90=False,
    align_centers=True,
):
    """Align coordinates with flipping and rotation, then translation.

    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames containing position data
    x_col, y_col : str
        Column names for x and y coordinates
    reference_df : int (1 or 2)
        Which dataframe to use as reference (the other will be transformed)
    flip_x : bool
        Whether to flip x coordinates (negate x values)
    flip_y : bool
        Whether to flip y coordinates (negate y values)
    rotate_90 : bool
        Whether to rotate 90 degrees counterclockwise
    align_centers : bool
        Whether to translate to align centers with reference dataset

    Returns:
    --------
    df1_aligned, df2_aligned : pandas.DataFrame
        Aligned dataframes with modified coordinates
    transformation_info : dict
        Information about the transformation applied
    """
    df1_aligned = df1.copy()
    df2_aligned = df2.copy()

    # Calculate initial centers
    center1_orig = (df1[x_col].mean(), df1[y_col].mean())
    center2_orig = (df2[x_col].mean(), df2[y_col].mean())

    if reference_df == 1:
        # Transform df2 to match df1
        target_df = df2_aligned
        target_coords = df2[[x_col, y_col]].values
        reference_center = center1_orig
        transform_center_orig = center2_orig
    else:
        # Transform df1 to match df2
        target_df = df1_aligned
        target_coords = df1[[x_col, y_col]].values
        reference_center = center2_orig
        transform_center_orig = center1_orig

    # Get the center of the dataset being transformed
    transform_center = np.mean(target_coords, axis=0)

    # Center the coordinates around origin
    centered_coords = target_coords - transform_center

    # Step 1: Flip coordinates if requested
    if flip_x and flip_y:
        centered_coords[:, 0] = -centered_coords[:, 0]
        centered_coords[:, 1] = -centered_coords[:, 1]
        print(f"Step 1: Flipped X and Y coordinates")
    elif flip_x:
        centered_coords[:, 0] = -centered_coords[:, 0]
        print(f"Step 1: Flipped X coordinates")
    elif flip_y:
        centered_coords[:, 1] = -centered_coords[:, 1]
        print(f"Step 1: Flipped Y coordinates")
    else:
        print(f"Step 1: No flip applied")

    # Step 2: Rotate 90 degrees counterclockwise if requested
    if rotate_90:
        # 90-degree counterclockwise rotation matrix: [[0, -1], [1, 0]]
        # New x = -old y, New y = old x
        new_coords = np.zeros_like(centered_coords)
        new_coords[:, 0] = -centered_coords[:, 1]  # new x = -old y
        new_coords[:, 1] = centered_coords[:, 0]  # new y = old x
        centered_coords = new_coords
        print(f"Step 2: Rotated 90 degrees counterclockwise")
    else:
        print(f"Step 2: No rotation applied")

    # Step 3: Calculate translation
    if align_centers:
        transformed_center = np.mean(centered_coords, axis=0)
        translation = np.array(reference_center) - transformed_center
        # Apply translation
        centered_coords = centered_coords + translation
        print(f"Step 3: Aligned with reference center")
    else:
        translation = transform_center
        # Apply translation back to original center
        centered_coords = centered_coords + translation
        print(f"Step 3: No alignment applied")

    # Update the target dataframe
    target_df[x_col] = centered_coords[:, 0]
    target_df[y_col] = centered_coords[:, 1]

    # Verify final centers
    final_center1 = (df1_aligned[x_col].mean(), df1_aligned[y_col].mean())
    final_center2 = (df2_aligned[x_col].mean(), df2_aligned[y_col].mean())

    if reference_df == 1:
        final_center = final_center2
    else:
        final_center = final_center1

    transformation_info = {
        "flip_x": flip_x,
        "flip_y": flip_y,
        "rotate_90": rotate_90,
        "align_centers": align_centers,
        "translation": translation if align_centers else transform_center,
        "reference_df": reference_df,
        "original_centers": (center1_orig, center2_orig),
        "final_centers": (final_center1, final_center2),
    }

    return df1_aligned, df2_aligned, transformation_info


def find_closest_tiles(sbs_metadata, ph_metadata, sbs_tile_id, verbose=True):
    """Find closest tiles in ph_metadata to a specific tile in sbs_metadata.

    Args:
        sbs_metadata: DataFrame with x_pos, y_pos columns
        ph_metadata: DataFrame with x_pos, y_pos columns
        sbs_tile_id: ID of sbs tile to find neighbors for
        verbose: If True, print top 3 matches

    Returns:
        DataFrame of ph tiles sorted by distance to sbs_tile_id
    """
    # Get sbs tile coordinates
    sbs_tile = sbs_metadata[sbs_metadata.tile == sbs_tile_id]
    sbs_x, sbs_y = sbs_tile["x_pos"].iloc[0], sbs_tile["y_pos"].iloc[0]

    # Calculate distances to all ph tiles
    distances = np.sqrt(
        (ph_metadata["x_pos"] - sbs_x) ** 2 + (ph_metadata["y_pos"] - sbs_y) ** 2
    )

    # Return sorted results
    result = ph_metadata.copy()
    result["distance"] = distances

    if verbose:
        # Print the top 3 closest tiles
        closest_tiles = result.nsmallest(3, "distance")
        print(f"\nTop 3 closest tiles to SBS tile {sbs_tile_id}:")
        for idx, row in closest_tiles.iterrows():
            print(f"  Tile {row['tile']}: Distance = {row['distance']:.2f}")

    return result.sort_values("distance")


def filter_low_score_seeds(pairs_df, score_col="score", k=3.0, min_keep=5):
    """Drop initial-site seeds whose score is a low outlier relative to the cohort.

    The alignment score's absolute scale varies by screen, so this uses a robust relative
    cut (median and MAD) rather than a fixed floor: a seed is dropped only if its score is
    more than `k` robust standard deviations below the median. At least `min_keep` seeds are
    always retained (falling back to the top-scoring ones) so filtering can never push a well
    below the minimum the pipeline requires.

    Args:
        pairs_df (pandas.DataFrame): Candidate seed pairs with a score column.
        score_col (str, optional): Name of the score column. Defaults to "score".
        k (float, optional): Number of robust standard deviations below the median at which a
            seed is considered a low outlier. Defaults to 3.0.
        min_keep (int, optional): Minimum number of seeds to retain. Defaults to 5.

    Returns:
        pandas.DataFrame: The retained seed pairs.
    """
    if len(pairs_df) <= min_keep:
        return pairs_df

    scores = pairs_df[score_col].to_numpy()
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    if mad == 0:
        return pairs_df

    threshold = median - k * 1.4826 * mad
    kept = pairs_df[pairs_df[score_col] >= threshold]

    # Never filter below the minimum the pipeline needs — keep the best-scoring seeds instead
    if len(kept) < min_keep:
        kept = pairs_df.sort_values(score_col, ascending=False).head(min_keep)
    return kept


def fast_merge_example(
    ph_tile,
    sbs_site,
    alignment_df,
    phenotype_info,
    sbs_info,
    threshold,
    local_refinement=None,
    warp_kwargs=None,
):
    """Process and plot PH tile and SBS site pairs."""
    print(f"\nProcessing PH tile {ph_tile} and SBS site {sbs_site}...")

    # Get alignment vector
    alignment_vec = alignment_df[
        (alignment_df["tile"] == ph_tile) & (alignment_df["site"] == sbs_site)
    ]

    # Validation checks
    if alignment_vec.empty:
        print(f"  No valid alignment found")
        return False

    alignment_vec = alignment_vec.iloc[0]

    if not hasattr(alignment_vec.get("rotation"), "ndim") or not hasattr(
        alignment_vec.get("translation"), "ndim"
    ):
        print(f"  Invalid alignment data")
        print(f"  Rotation: {alignment_vec.get('rotation')}")
        print(f"  Translation: {alignment_vec.get('translation')}")
        return False

    # Try plotting
    try:
        plot_merge_example(
            phenotype_info,
            sbs_info,
            alignment_vec,
            threshold=threshold,
            local_refinement=local_refinement,
            warp_kwargs=warp_kwargs,
        )
        return True
    except Exception as e:
        print(f"  Error plotting: {str(e)}")
        return False
