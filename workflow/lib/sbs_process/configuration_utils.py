"""Utility functions for configuring SBS process parameters."""

import numpy as np

from lib.shared.configuration_utils import outline_mask


def annotate_segment_on_sequencing_data(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on sequencing data.

    This function overlays outlines of nuclei and cells on the provided sequencing data.

    Args:
        data (numpy.ndarray): Sequencing data with shape (cycles, channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated sequencing data with outlines of nuclei and cells.
    """
    # Ensure data has at least 4 dimensions
    if data.ndim == 3:
        data = data[None]

    # Get dimensions of the sequencing data
    cycles, channels, height, width = data.shape

    # Create an array to store annotated data
    annotated = np.zeros((cycles, channels + 1, height, width), dtype=np.uint16)

    # Generate combined mask for nuclei and cells outlines
    mask = (outline_mask(nuclei, direction="inner") > 0) + (
        outline_mask(cells, direction="inner") > 0
    )

    # Copy original data to annotated data
    annotated[:, :channels] = data

    # Add combined mask to the last channel
    annotated[:, channels] = mask

    return np.squeeze(annotated)
