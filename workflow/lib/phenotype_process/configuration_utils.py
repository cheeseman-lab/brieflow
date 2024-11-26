"""Utility functions for configuring phenotype process parameters."""

import numpy as np

from lib.shared.configuration_utils import outline_mask


def annotate_on_phenotyping_data(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on phenotyping data.

    This function overlays outlines of nuclei and cells on the provided phenotyping data.

    Args:
        data (numpy.ndarray): Phenotyping data with shape (channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated phenotyping data with outlines of nuclei and cells.
    """
    # Ensure data has at least 3 dimensions
    if data.ndim == 2:
        data = data[None]

    # Get dimensions of the phenotyping data
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
