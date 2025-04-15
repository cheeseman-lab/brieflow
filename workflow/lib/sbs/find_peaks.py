"""Find SBS peaks."""

import numpy as np
import matplotlib.pyplot as plt

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
