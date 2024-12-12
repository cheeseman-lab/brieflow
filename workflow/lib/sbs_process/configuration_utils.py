"""Utility functions for configuring SBS process parameters."""

import numpy as np

from lib.shared.configuration_utils import outline_mask


def annotate_segment_on_sequencing_data(cellpose_rgb, nuclei, cells):
    """Annotate outlines of nuclei and cells on sequencing data.

    This function overlays outlines of nuclei and cells on the provided cellpose_rgb data.

    Args:
        cellpose_rgb (numpy.ndarray): CellPose-ready RGB data with shape (channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated sequencing data with outlines of nuclei and cells.
    """
    # Ensure data has at 3 dimensions
    if cellpose_rgb.ndim != 3:
        raise ValueError("Data must have 3 dimensions")

    # Get dimensions of the sequencing data
    channels, height, width = cellpose_rgb.shape

    # Create an array to store annotated data with an extra channel for the outlines
    annotated = np.zeros((channels + 1, height, width), dtype=np.uint16)

    # Copy original data to the first 'channels' slots of the annotated data
    annotated[:channels] = cellpose_rgb

    # Generate combined mask for nuclei and cells outlines
    mask = (outline_mask(nuclei, direction="inner") > 0) | (
        outline_mask(cells, direction="inner") > 0
    )

    # Add combined mask to the last channel of annotated data
    annotated[channels] = mask.astype(np.uint16)

    return np.squeeze(annotated)
