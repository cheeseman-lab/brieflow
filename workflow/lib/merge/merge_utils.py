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


from lib.merge.fast_merge import build_linear_model, refine_local_warp, match_cells
from lib.shared.alignment_overlay import magenta_green_overlay


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
    # Create the figure — two panels sharing one matched/unmatched coloring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

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

    # Panel 2: panel 1 without residuals; plot Y_pred, never a min-max rescale of X
    ax2.scatter(
        Y[:, 0], Y[:, 1], c="lightgray", s=12, alpha=0.15, label=f"SBS field ({n_sbs})"
    )
    ax2.scatter(
        Y_pred[matched_ph_mask, 0],
        Y_pred[matched_ph_mask, 1],
        c="#2f6fb0",
        s=14,
        alpha=0.35,
        label=f"matched phenotype ({n_matched})",
    )
    ax2.scatter(
        Y_pred[~matched_ph_mask, 0],
        Y_pred[~matched_ph_mask, 1],
        marker="*",
        c="#e8b93a",
        s=60,
        alpha=0.9,
        label=f"unmatched phenotype ({n_unmatched})",
    )
    ax2.set_aspect("equal")
    ax2.set_title("Matched vs unmatched phenotype (SBS pixel space)")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


def load_merge_dapi_pair(
    root_fp, plate, well, ph_tile, sbs_site, image_format, ph_channels, sbs_channels
):
    """Load the SBS and phenotype DAPI images of one tile-site pair for the merge overlay.

    Reads the aligned images (first SBS cycle) and falls back to the illumination-corrected
    images when the aligned ones were not kept.

    Args:
        root_fp (str | Path): Brieflow output root.
        plate (int | str): Plate.
        well (str): Well, e.g. "A1".
        ph_tile (int): Phenotype tile.
        sbs_site (int): SBS site (tile).
        image_format (str): "tiff" or "zarr".
        ph_channels (list[str]): Phenotype channel names; DAPI is used, else the first.
        sbs_channels (list[str]): SBS channel names; DAPI is used, else the first.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: (sbs_dapi, ph_dapi); None for an image
            that is not found.
    """
    from lib.shared.file_utils import get_image_output_path
    from lib.shared.image_io import read_image

    def _dapi(module, tile, channels):
        location = {"plate": plate, "well": well, "tile": tile}
        index = channels.index("DAPI") if "DAPI" in channels else 0
        for name in ("aligned", "illumination_corrected"):
            path = (
                Path(root_fp)
                / module
                / get_image_output_path(location, name, image_format)
            )
            if path.exists():
                image = read_image(path)
                return image.reshape(-1, *image.shape[-2:])[index]
        return None

    return (
        _dapi("sbs", sbs_site, list(sbs_channels)),
        _dapi("phenotype", ph_tile, list(ph_channels)),
    )


def plot_merge_alignment_overlay(sbs_dapi, ph_dapi, alignment_df, ph_tile, sbs_site):
    """Overlay phenotype DAPI mapped into SBS pixel space (green) on SBS DAPI (magenta).

    The phenotype DAPI image is resampled through the tile-site affine model the merge
    uses. The left panel shows the whole SBS site for context: SBS DAPI in magenta outside
    the outlined phenotype tile (SBS-only signal) and the overlay inside it; the right panel
    zooms on the tile. Inside the tile both images are brightness-matched for display (see
    `magenta_green_overlay`): nuclei in both images read white or grey, a shift leaves magenta
    and green fringes, and a nucleus found in one image only stays fully magenta or green.
    The zoom's title gives the residual shift (dy, dx SBS px) of the mapped phenotype image.

    Args:
        sbs_dapi (np.ndarray): 2D SBS DAPI image of the site.
        ph_dapi (np.ndarray): 2D phenotype DAPI image of the tile.
        alignment_df (pandas.DataFrame): Initial alignment with tile, site, rotation and
            translation columns.
        ph_tile (int): Phenotype tile.
        sbs_site (int): SBS site (tile).

    Returns:
        matplotlib.figure.Figure: The figure, or None if the pair has no alignment.
    """
    from skimage.registration import phase_cross_correlation

    match = alignment_df[
        (alignment_df["tile"] == ph_tile) & (alignment_df["site"] == sbs_site)
    ]
    if match.empty:
        print(f"  No alignment for PH tile {ph_tile} and SBS site {sbs_site}")
        return None
    rotation = np.asarray(match.iloc[0]["rotation"], dtype=float)
    translation = np.asarray(match.iloc[0]["translation"], dtype=float)

    ph_in_sbs = map_phenotype_to_sbs(ph_dapi, sbs_dapi.shape, rotation, translation)
    footprint = (
        map_phenotype_to_sbs(
            np.ones(ph_dapi.shape, dtype=np.float32),
            sbs_dapi.shape,
            rotation,
            translation,
        )
        > 0.5
    )
    rows, cols = (
        np.flatnonzero(footprint.any(axis=1)),
        np.flatnonzero(footprint.any(axis=0)),
    )
    box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    inside = footprint[box]
    sbs, ph = (_fill_outside(image[box], inside) for image in (sbs_dapi, ph_in_sbs))

    shift = phase_cross_correlation(
        np.where(inside, ph, 0),
        np.where(inside, sbs, 0),
        upsample_factor=2,
        normalization=None,
    )[0]
    overlay = magenta_green_overlay(sbs, ph) * inside[..., None]

    # context: SBS DAPI in magenta (SBS only), with the overlay pasted inside the footprint
    low, high = np.percentile(sbs_dapi, (1, 99.5))
    sbs_scaled = np.clip(
        (sbs_dapi.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1
    )
    context = np.stack([sbs_scaled, np.zeros_like(sbs_scaled), sbs_scaled], axis=-1)
    context[box] = np.where(inside[..., None], overlay, context[box])
    corners = np.array(
        [[0, 0], [0, ph_dapi.shape[1]], ph_dapi.shape, [ph_dapi.shape[0], 0], [0, 0]],
        dtype=float,
    )
    outline = corners @ rotation.T + translation

    fig, (ax_context, ax_zoom) = plt.subplots(
        1, 2, figsize=(13, 7), layout="constrained"
    )
    ax_context.imshow(context, interpolation="nearest")
    ax_context.plot(outline[:, 1], outline[:, 0], color="yellow", lw=0.8)
    ax_context.add_patch(
        mpatches.Rectangle(
            (box[1].start, box[0].start),
            box[1].stop - box[1].start,
            box[0].stop - box[0].start,
            fill=False,
            edgecolor="cyan",
            lw=0.8,
            ls="--",
        )
    )
    ax_context.set_title(
        "SBS site: magenta outside the yellow phenotype tile = SBS only; cyan = zoom",
        fontsize=10,
    )
    ax_zoom.imshow(overlay, interpolation="nearest")
    ax_zoom.set_title(
        f"zoom: residual shift ({shift[0]:+.1f}, {shift[1]:+.1f}) SBS px", fontsize=10
    )
    for ax in (ax_context, ax_zoom):
        ax.axis("off")
    fig.suptitle(
        f"PH tile {ph_tile} → SBS site {sbs_site}. Inside the phenotype tile: SBS DAPI "
        "magenta, phenotype DAPI green;\nwhite/grey = nucleus in both (aligned), "
        "magenta/green fringes = shift, fully magenta or green = in one image only",
        fontsize=10,
    )
    plt.show()
    return fig


def map_phenotype_to_sbs(ph_image, sbs_shape, rotation, translation):
    """Resample a phenotype image into SBS pixel space with the merge's affine model.

    Args:
        ph_image (np.ndarray): 2D phenotype image.
        sbs_shape (tuple[int, int]): Shape of the SBS image.
        rotation (np.ndarray): 2x2 matrix of the model (phenotype (i, j) to SBS (i, j)).
        translation (np.ndarray): Translation of the model, in SBS pixels.

    Returns:
        np.ndarray: Phenotype image in SBS pixel space, 0 outside the phenotype tile.
    """
    from scipy import ndimage

    # the merge model maps phenotype (i, j) to SBS (i, j) as X @ rotation.T + translation
    rows, cols = np.indices(sbs_shape)
    sbs_coords = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(float)
    ph_coords = (sbs_coords - np.asarray(translation, dtype=float)) @ np.linalg.inv(
        np.asarray(rotation, dtype=float).T
    )
    return ndimage.map_coordinates(
        np.asarray(ph_image, dtype=np.float32), ph_coords.T, order=1, cval=0.0
    ).reshape(sbs_shape)


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


def _fill_outside(image, inside):
    """Image with pixels outside the footprint set to the footprint's median, for display."""
    image = np.asarray(image, dtype=np.float32).copy()
    image[~inside] = np.median(image[inside])
    return image
