"""Find SBS peaks!

Find peaks of signal in SBS data using either a local maxima of base channel standard deviation
(standard approach) or a deep learning model (Spotiflow).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from spotiflow.model import Spotiflow

from lib.shared.image_utils import remove_channels


def find_peaks(standard_deviation_data, width=5, remove_index=None):
    """Find local maxima and label by difference to next-highest neighboring pixel.

    Conventionally used to estimate SBS read locations by inputting the standard deviation score.

    Args:
        standard_deviation_data (numpy.ndarray): 2D image data of sbs standard deviation.
        width (int, optional): Neighborhood size for finding local maxima. Default is 5.
        remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

    Returns:
        peaks (numpy.ndarray): Local maxima scores, dimensions same as data. At a maximum, the value is max - min in the defined neighborhood, elsewhere zero.
    """
    # Remove specified index channel if needed
    if remove_index is not None:
        standard_deviation_data = remove_channels(standard_deviation_data, remove_index)

    # If data is 2D, convert it to a list
    if standard_deviation_data.ndim == 2:
        standard_deviation_data = [standard_deviation_data]

    # Find peaks in each image with a defined neighborhood size
    peaks = [
        find_neighborhood_peaks(x, n=width) if x.max() > 0 else x
        for x in standard_deviation_data
    ]

    # Convert the list of peaks to a numpy array and squeeze it to remove singleton dimensions
    peaks = np.array(peaks).squeeze()

    return peaks


def find_neighborhood_peaks(data, n=5):
    """Finds local maxima in the input data.

    At a maximum, the value is max - min in a neighborhood of width `n`.
    Elsewhere, it is zero.

    Args:
        data (numpy.ndarray): Input data.
        n (int, optional): Width of the neighborhood for finding local maxima. Default is 5.

    Returns:
        peaks (numpy.ndarray): Local maxima scores.
    """
    # Import necessary modules and functions
    from scipy import ndimage as ndi
    import numpy as np

    # Define the maximum and minimum filters for finding local maxima
    filters = ndi.filters

    # Define the neighborhood size based on the input data dimensions
    neighborhood_size = (1,) * (data.ndim - 2) + (n, n)

    # Apply maximum and minimum filters to the data to find local maxima
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)

    # Calculate the difference between maximum and minimum values to identify peaks
    peaks = data_max - data_min

    # Set values to zero where the original data is not equal to the maximum values
    peaks[data != data_max] = 0

    # Remove peaks close to the edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0

    return peaks


def find_peaks_spotiflow(
    aligned_images,
    cycle_idx=0,
    model="general",
    prob_thresh=0.5,
    min_distance=3,
    subpixel_precision=True,
    remove_index=None,
    verbose=True,
    round_coords=True,
):
    """Detect SBS peaks using the Spotiflow deep learning model.

    Processes each base channel separately and combines the results.

    Args:
        aligned_images (numpy.ndarray): Aligned SBS images with shape
            (cycles, channels, height, width).
        cycle_idx (int, optional): Index of the cycle to use for peak detection.
            Defaults to 0.
        model (str or spotiflow.model.Spotiflow, optional): Either a model name to
            load from pretrained models or a Spotiflow model instance.
            Defaults to "general".
        prob_thresh (float, optional): Probability threshold for spot detection.
            Defaults to 0.5.
        min_distance (int, optional): Minimum distance between spots in pixels.
            Defaults to 3.
        subpixel_precision (bool, optional): Whether to use subpixel precision
            for spot detection. Defaults to True.
        remove_index (int or None, optional): Index of channel to remove from
            analysis, typically non-SBS channels like DAPI. Defaults to None.
        verbose (bool, optional): Whether to print progress information.
            Defaults to True.
        round_coords (bool, optional): Whether to round coordinates to integers.
            Defaults to True.

    Returns:
        tuple:
            - peaks (numpy.ndarray): Binary array of shape (height, width) where 1
              indicates a peak location and 0 indicates no peak.
            - all_base_coords (list): List of peak coordinates for each base channel.
    """
    # Load model if string is provided
    if isinstance(model, str):
        if verbose:
            print(f"Loading Spotiflow '{model}' model...")
        model = Spotiflow.from_pretrained(model)

    if verbose:
        print(f"Detecting peaks for each base channel using cycle {cycle_idx}...")

    # Get spots for cycle cycle_idx
    spots = aligned_images[cycle_idx, :, :, :]

    # Remove specified index channel if needed
    if remove_index is not None:
        spots = remove_channels(spots, remove_index)

    # Get dimensions
    n_bases, height, width = spots.shape

    # Initialize output array
    peaks = np.zeros((height, width), dtype=np.int8)

    # List to store coordinates for each base
    all_base_coords = []

    # Process each base channel separately
    for base_idx in range(n_bases):
        if verbose:
            print(f"Processing base channel {base_idx + 1}/{n_bases}...")

        # Extract data for current base
        base_data = spots[base_idx, :, :]

        # Use the base_data directly - no normalization
        # Run Spotiflow spot detection
        peak_coords, _ = model.predict(
            base_data,
            prob_thresh=prob_thresh,
            min_distance=min_distance,
            subpix=subpixel_precision,
            verbose=False,
        )

        # Round coordinates if requested
        if round_coords:
            peak_coords = np.array(
                [(int(y), int(x)) for y, x in zip(peak_coords[:, 0], peak_coords[:, 1])]
            )

        # Filter out coordinates outside image boundaries
        valid_peaks = (
            (peak_coords[:, 0] >= 0)
            & (peak_coords[:, 0] < height)
            & (peak_coords[:, 1] >= 0)
            & (peak_coords[:, 1] < width)
        )
        valid_coords = peak_coords[valid_peaks]

        if verbose:
            print(f"  Base {base_idx + 1}: {len(valid_coords)} spots detected")

        # Store coordinates for this base
        all_base_coords.append(valid_coords)

    # Combine results from all bases while enforcing minimum distance
    if verbose:
        print("Combining results from all bases...")

    # First, collect all coordinates
    all_coords = (
        np.vstack(all_base_coords)
        if all_base_coords and all(len(coords) > 0 for coords in all_base_coords)
        else np.empty((0, 2))
    )

    if len(all_coords) > 0:
        # Remove duplicates (exact same positions)
        all_coords = np.unique(all_coords, axis=0)

        # Initialize list for final coordinates
        final_coords = []

        # Sort coordinates by intensity if available
        # For simplicity, we'll just process them in order
        remaining_coords = all_coords.copy()

        while len(remaining_coords) > 0:
            # Take the first coordinate as a "seed"
            seed_coord = remaining_coords[0]
            final_coords.append(seed_coord)

            # Calculate distances to all remaining coordinates
            distances = cdist([seed_coord], remaining_coords)

            # Find coordinates that are far enough from the seed
            far_enough = distances[0] > min_distance

            # Update remaining coords to only those far enough from the seed
            remaining_coords = remaining_coords[far_enough]

        # Convert to numpy array
        final_coords = np.array(final_coords)

        # Create binary peak array
        peaks = np.zeros((height, width), dtype=np.int8)
        peaks[final_coords[:, 0], final_coords[:, 1]] = 1

        if verbose:
            print(
                f"Final result: {len(final_coords)} spots after enforcing minimum distance of {min_distance}"
            )
    else:
        if verbose:
            print("No spots detected in any channel")
        final_coords = np.empty((0, 2))

    return peaks, all_base_coords


def plot_channels_with_peaks(
    maxed_data,
    peaks_array,
    bases,
    cycle_number=0,
    threshold_peaks=None,
    peak_colors=None,
    peak_labels=None,
    figsize=(12, 12),
):
    """Plot individual channel data with detected peaks overlaid.

    Args:
        maxed_data : numpy.ndarray
            Max filtered data with shape (cycles, channels, height, width)
            or (channels, height, width)
        peaks_array : numpy.ndarray
            2D array where values indicate peaks (binary or intensity values)
        bases : list of str
            List of base names (e.g., ['G', 'T', 'A', 'C'])
        cycle_number : int
            Cycle number for subsetting the data and for the title
        threshold_peaks : float, optional
            Threshold value to consider a point as a peak.
            If None, treats peaks_array as binary (non-zero values are peaks).
        peak_colors : list of str, optional
            Colors for the peaks. Defaults to ['orange']
        peak_labels : list of str, optional
            Labels for the peaks in the legend. Defaults to None
        figsize : tuple, optional
            Figure size. Defaults to (12, 12)
    """
    # Default values
    if peak_colors is None:
        peak_colors = ["orange"]
    if peak_labels is None:
        peak_labels = ["Detected Peaks"]

    # Standard colormaps for bases
    standard_cmaps = ["Greens", "Reds", "Blues", "Purples"]

    # Extract data for the specified cycle
    if len(maxed_data.shape) == 4:  # (cycles, channels, height, width)
        cycle_data = maxed_data[cycle_number]
    else:  # (channels, height, width)
        cycle_data = maxed_data

    # Convert peaks array to coordinate list
    if threshold_peaks is not None:
        peak_coords = np.argwhere(peaks_array > threshold_peaks)
        threshold_text = f" (threshold={threshold_peaks})"
        print(f"Found {len(peak_coords)} peaks above threshold {threshold_peaks}")
    else:
        # If no threshold provided, assume binary array (non-zero values are peaks)
        peak_coords = np.argwhere(peaks_array > 0)
        threshold_text = ""
        print(f"Found {len(peak_coords)} peaks (binary array, non-zero values)")

    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Cycle {cycle_number} with Detected Spots{threshold_text}", fontsize=16
    )
    axes = axes.flatten()

    # Plot each base channel with spots
    for i, base in enumerate(bases):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            # Get the channel data
            channel_data = cycle_data[i]

            # Apply percentile-based scaling
            vmin = np.percentile(channel_data, 1)
            vmax = np.percentile(channel_data, 99.5)

            # Display the channel image
            im = axes[i].imshow(
                channel_data, cmap=standard_cmaps[i], vmin=vmin, vmax=vmax
            )

            # Add peaks
            axes[i].scatter(
                peak_coords[:, 1],
                peak_coords[:, 0],
                facecolors="none",
                edgecolors=peak_colors[0],
                s=15,
                linewidths=0.5,
            )

            # Set title and formatting
            axes[i].set_title(f"{base} Channel")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Add legend with peak count
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=peak_colors[0],
            markersize=8,
            label=f"{peak_labels[0]} ({len(peak_coords)})",
        )
    ]
    fig.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    return fig, axes
