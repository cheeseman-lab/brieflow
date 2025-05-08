"""Shared utilties for configuring Brieflow process parameters.

This includes:
- Header string for Brieflow config file.
- Function to create the Brieflow samples dataframe with file location and metadata.
- Functions for displaying SBS/phenotype images and segmentations.
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
import skimage.morphology
from scipy.spatial.distance import cdist

from lib.merge.merge import build_linear_model

CONFIG_FILE_HEADER = """
# ~BrieFlow analysis configuration file~

# All paths are resolved relative to the directory the workflow is run from.

# Parameters:
"""


def create_samples_df(images_fp, sample_pattern, metadata, metadata_order_type):
    """Generate samples dataframe from a directory of images.

    Samples dataframe includes path to images and image metadata extracted from file names.
    The function reorders columns, changes column types, and sorts the DataFrame.

    Args:
        images_fp (Path): Path to the directory containing images.
        sample_pattern (str): Regular expression pattern to extract metadata from image file names.
        metadata (list): List of metadata keys to extract from the file names.
        metadata_order_type (dict): Dictionary specifying the order and data types of metadata columns.

    Returns:
        DataFrame: DataFrame containing sample metadata and file paths.
    """
    if images_fp is None:
        print("No image directory provided, returning an empty sample DataFrame!")
        return pd.DataFrame(columns=["sample_fp"] + metadata)

    samples_data = []
    sample_regex = re.compile(sample_pattern)

    # Find and extract metadata from matching files
    for image_fp in Path(images_fp).rglob("*"):
        match = sample_regex.search(str(image_fp))
        if match:
            samples_data.append(
                {"sample_fp": str(image_fp), **dict(zip(metadata, match.groups()))}
            )

    # Create DataFrame
    samples_df = pd.DataFrame(samples_data)

    if samples_df.empty:
        raise ValueError(
            f"No matching files found in {images_fp} with pattern {sample_pattern}"
        )

    # Convert column types according to metadata_order_type
    for column, column_type in metadata_order_type.items():
        if column in samples_df.columns:
            samples_df[column] = samples_df[column].astype(column_type)

    # Reorder columns
    column_order = ["sample_fp"] + list(metadata_order_type.keys())
    samples_df = samples_df[column_order]

    # Sort DataFrame by metadata columns
    sort_columns = list(metadata_order_type.keys())
    samples_df = samples_df.sort_values(by=sort_columns)

    # Reset index
    samples_df.reset_index(drop=True, inplace=True)

    return samples_df


def create_micropanel(microimages, num_cols=2, figscaling=6, add_channel_label=True):
    """Creates a Micropanel from a list of Microimages.

    Dynamically arranges microimages into a grid based on the specified number of columns.

    Args:
        microimages (list): A list of Microimage objects to be displayed in the panel.
        num_cols (int, optional): Number of columns in the grid. Defaults to 2.
        figscaling (int, optional): Scaling factor for the figure size. Defaults to 4.
        add_channel_label (bool, optional): If True, adds channel labels to the microimages. Defaults to True

    Returns:
        Micropanel: A Micropanel object with microimages arranged in a grid.
    """
    # Calculate grid dimensions
    num_images = len(microimages)
    num_rows = math.ceil(num_images / num_cols)

    # Create panel with dynamic rows
    panel = Micropanel(rows=num_rows, cols=num_cols, figscaling=figscaling)

    # Add all microimages to the panel
    for i, microimage in enumerate(microimages):
        row = i // num_cols
        col = i % num_cols
        panel.add_element([row, col], microimage)

    # Add channel labels to the microimages
    if add_channel_label:
        panel.add_channel_label()

    return panel


def random_cmap(alpha=0.5, num_colors=256):
    """Create a random colormap for segmentation.

    Args:
        alpha (float, optional): Transparency value for the colors in the colormap, ranging from 0 (transparent)
            to 1 (opaque). Defaults to 0.5.
        num_colors (int, optional): Number of colors to generate in the colormap. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: A colormap object with randomly generated colors, where the first
            color is set to black with full transparency.
    """
    colmat = np.random.rand(num_colors, 4)
    colmat[:, -1] = alpha
    # Set the first color to black with full transparency
    colmat[0, :] = [0, 0, 0, 1]
    cmap = matplotlib.colors.ListedColormap(colmat)
    return cmap


def outline_mask(arr, direction="outer", width=1):
    """Remove interior of label mask in `arr`.

    Args:
        arr (numpy.ndarray): The input label mask array.
        direction (str, optional): The direction of outlining. 'outer' outlines the outer boundary, 'inner' outlines the inner boundary. Default is 'outer'.
        width (int, optional): The width of the structuring element used for erosion and dilation. Default is 1.

    Returns:
        numpy.ndarray: The label mask array with the outlined interior removed.

    Raises:
        ValueError: If `direction` is neither 'outer' nor 'inner'.
    """
    selem = skimage.morphology.disk(
        width
    )  # Create a disk-shaped structuring element with the specified width
    arr = (
        arr.copy()
    )  # Create a copy of the input array to avoid modifying the original array
    if direction == "outer":  # If outlining direction is 'outer'
        mask = skimage.morphology.erosion(
            arr, selem
        )  # Erode the mask using the structuring element
        arr[mask > 0] = 0  # Set interior pixels to 0
        return arr  # Return the modified array
    elif direction == "inner":  # If outlining direction is 'inner'
        mask1 = (
            skimage.morphology.erosion(arr, selem) == arr
        )  # Create a mask for pixels on the inner boundary
        mask2 = (
            skimage.morphology.dilation(arr, selem) == arr
        )  # Create a mask for pixels on the outer boundary
        arr[mask1 & mask2] = (
            0  # Set pixels within the inner boundary and outside the outer boundary to 0
        )
        return arr  # Return the modified array
    else:  # If direction is neither 'outer' nor 'inner'
        raise ValueError(direction)  # Raise a ValueError


def image_segmentation_annotations(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on image data.

    This function overlays outlines of nuclei and cells on the provided image data.

    Args:
        data (numpy.ndarray): Image data with shape (channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated image data with outlines of nuclei and cells.
    """
    # Ensure data has at least 3 dimensions
    if data.ndim == 2:
        data = data[None]

    # Get dimensions of the image data
    channels, height, width = data.shape

    # Create an array to store annotated data
    annotated = np.zeros((channels + 1, height, width), dtype=np.uint16)

    # Generate combined mask for nuclei and cells outlines
    mask = (outline_mask(nuclei, direction="inner") > 0) + (
        outline_mask(cells, direction="inner") > 0
    )

    # Copy original data to annotated data
    annotated[:channels] = data

    # Add combined mask to the last channel
    annotated[channels] = mask

    return np.squeeze(annotated)


def plot_combined_tile_grid(ph_test_metadata, sbs_test_metadata):
    """Plots a combined grid of X-Y positions for PH and SBS datasets with annotations.

    Note: Plot sizing is hard coded with arbitrary values that will not work for ND2 images with different sizes.

    Args:
        ph_test_metadata (pd.DataFrame): DataFrame containing PH metadata with columns:
            'x_pos', 'y_pos', 'tile', and other relevant fields.
        sbs_test_metadata (pd.DataFrame): DataFrame containing SBS metadata with columns:
            'x_pos', 'y_pos', 'tile', and other relevant fields.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Create figure
    fig = plt.figure(figsize=(30, 24))

    # Scatter plot for PH data
    plt.scatter(
        ph_test_metadata["x_pos"],
        ph_test_metadata["y_pos"],
        s=450,
        c="white",
        marker="s",
        edgecolors="black",
        linewidths=1,
        alpha=0.7,
        label="PH",
    )

    # Label each PH point with the 'tile' variable
    for i, txt in enumerate(ph_test_metadata["tile"]):
        plt.annotate(
            txt,
            (ph_test_metadata["x_pos"].iloc[i], ph_test_metadata["y_pos"].iloc[i]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=12,
            color="black",
        )

    # Scatter plot for SBS data
    plt.scatter(
        sbs_test_metadata["x_pos"],
        sbs_test_metadata["y_pos"],
        s=1800,
        c="red",
        marker="s",
        edgecolors="black",
        linewidths=1,
        alpha=0.5,
        label="SBS",
    )

    # Label each SBS point with the 'tile' variable
    for i, txt in enumerate(sbs_test_metadata["tile"]):
        plt.annotate(
            txt,
            (sbs_test_metadata["x_pos"].iloc[i], sbs_test_metadata["y_pos"].iloc[i]),
            textcoords="offset points",
            xytext=(0, -7),
            ha="center",
            fontsize=12,
            color="red",
        )

    # Set labels and title
    plt.xlabel("X Position", fontsize=30)
    plt.ylabel("Y Position", fontsize=30)
    plt.title(
        "Combined Grid Plot of X-Y Positions with Field of View Labels, SBS & PH",
        fontsize=30,
    )

    # Add legend
    plt.legend(fontsize=30)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_merge_example(df_ph, df_sbs, alignment_vec, threshold=2):
    """Visualizes the merge process for a single tile-site pair.

    Args:
        df_ph (pandas.DataFrame): Phenotype data with 'i', 'j' columns.
        df_sbs (pandas.DataFrame): SBS data with 'i', 'j' columns.
        alignment_vec (dict): Contains 'rotation' and 'translation' for alignment.
        threshold (float, optional): Distance threshold for matching points. Defaults to 2.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    # Filter for specific tile and site
    df_ph_filtered = df_ph[df_ph["tile"] == alignment_vec["tile"]]
    df_sbs_filtered = df_sbs[df_sbs["tile"] == alignment_vec["site"]]

    # Get coordinates
    X = df_ph_filtered[["i", "j"]].values
    Y = df_sbs_filtered[["i", "j"]].values

    # Build model and predict
    model = build_linear_model(alignment_vec["rotation"], alignment_vec["translation"])
    Y_pred = model.predict(X)

    # Calculate distances
    distances = cdist(Y, Y_pred, metric="sqeuclidean")
    ix = distances.argmin(axis=1)
    filt = np.sqrt(distances.min(axis=1)) < threshold

    # Filter out Y_pred based on filt
    Y_pred = Y_pred[ix[filt]]

    # Calculate statistics
    n_ph = len(X)
    n_sbs = len(Y)
    n_matched = len(Y_pred)

    # Plot 1: Original Scale
    ax1.scatter(
        X[:, 0], X[:, 1], c="blue", s=20, alpha=0.5, label=f"Phenotype ({n_ph} points)"
    )
    ax1.scatter(
        Y_pred[:, 0],
        Y_pred[:, 1],
        c="red",
        s=20,
        alpha=0.5,
        label=f"Aligned SBS ({n_matched}) points)",
    )
    ax1.scatter(
        Y[:, 0],
        Y[:, 1],
        c="green",
        s=20,
        alpha=0.5,
        label=f"Original SBS ({n_sbs} points)",
    )

    # Draw lines between matched points that pass threshold
    for i in range(len(Y)):
        if filt[i]:
            ax1.plot([X[ix[i], 0], Y[i, 0]], [X[ix[i], 1], Y[i, 1]], "k-", alpha=0.1)

    ax1.set_title(
        f"Original Scale View\nPH:{alignment_vec['tile']}, SBS:{alignment_vec['site']}"
    )
    ax1.legend()

    # Plot 2: Scale PH values to SBS axis
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # Get the range and minimum of aligned SBS points (Y_pred)
    Y_pred_range = Y_pred.max(axis=0) - Y_pred.min(axis=0)
    Y_pred_min = Y_pred.min(axis=0)

    # Scale and translate phenotype points to align with SBS field
    X_scaled = (X_norm * Y_pred_range) + Y_pred_min

    ax2.scatter(
        Y[:, 0],
        Y[:, 1],
        c="lightgray",
        s=20,
        alpha=0.1,
        label=f"SBS Field ({n_sbs} points)",
    )
    ax2.scatter(
        Y_pred[:, 0],
        Y_pred[:, 1],
        c="red",
        s=20,
        alpha=0.25,
        label=f"Aligned SBS ({n_matched} points)",
    )
    ax2.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c="blue",
        s=20,
        alpha=0.25,
        label=f"Phenotype ({n_ph} points)",
    )

    ax2.set_title("Normalized Scale For PH Points Relative to SBS")
    ax2.legend()

    # Plot 3: Scale PH values to SBS axis
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # Get the range and minimum of aligned SBS points (Y_pred)
    Y_pred_range = Y_pred.max(axis=0) - Y_pred.min(axis=0)
    Y_pred_min = Y_pred.min(axis=0)
    # Scale and translate phenotype points to align with SBS field
    X_scaled = (X_norm * Y_pred_range) + Y_pred_min
    # Find unmatched phenotype points
    matched_ph_ix = np.unique(ix[filt])
    unmatched_ph_mask = ~np.isin(np.arange(len(X)), matched_ph_ix)
    # Plot SBS field and aligned points
    ax3.scatter(
        Y[:, 0],
        Y[:, 1],
        c="lightgray",
        s=20,
        alpha=0.1,
        label=f"SBS Field ({n_sbs} points)",
    )
    ax3.scatter(
        Y_pred[:, 0],
        Y_pred[:, 1],
        c="red",
        s=20,
        alpha=0.25,
        label=f"Aligned SBS ({n_matched} points)",
    )
    # Plot matched phenotype points in blue
    ax3.scatter(
        X_scaled[~unmatched_ph_mask][:, 0],
        X_scaled[~unmatched_ph_mask][:, 1],
        c="blue",
        s=20,
        alpha=0.25,
        label=f"Matched Phenotype ({n_matched} points)",
    )
    # Plot unmatched phenotype points in yellow with star marker
    ax3.scatter(
        X_scaled[unmatched_ph_mask][:, 0],
        X_scaled[unmatched_ph_mask][:, 1],
        marker="*",
        c="yellow",
        s=100,
        alpha=1,
        label=f"Unmatched Phenotype ({sum(unmatched_ph_mask)} points)",
    )
    # Optionally add labels for unmatched points
    for i in np.where(unmatched_ph_mask)[0]:
        ax3.annotate(
            f"Cell {i}",
            (X_scaled[i, 0], X_scaled[i, 1]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )
    ax3.set_title(
        "Normalized Scale For PH Points Relative to SBS (with unmatched points)"
    )
    ax3.legend()

    plt.tight_layout()
    plt.show()
