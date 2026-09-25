"""Function for identifying and isolating the cytoplasm region."""

import numpy as np


def identify_cytoplasm_cellpose(nuclei, cells):
    """Identifies and isolates the cytoplasm region in an image based on the provided nuclei and cells masks.

    Args:
        nuclei (ndarray): A 2D array representing the nuclei regions.
        cells (ndarray): A 2D array representing the cells regions.

    Returns:
        ndarray: A 2D array representing the cytoplasm regions.

    Raises:
        ValueError: If the nuclei and cell masks are not reconciled (label counts differ).
    """
    # Each cell is paired with the same-label nucleus, so the masks must be reconciled
    if len(np.unique(nuclei)) != len(np.unique(cells)):
        raise ValueError(
            f"Cannot identify cytoplasms: {len(np.unique(nuclei)) - 1} nuclei vs "
            f"{len(np.unique(cells)) - 1} cells. Cytoplasm needs reconciled masks; set "
            "`reconcile` (e.g. 'contained_in_cells' or 'consensus') in the config."
        )

    # Create an empty cytoplasmic mask with the same shape as cells
    cytoplasms = np.zeros(cells.shape)

    # Iterate over each unique cell label
    for cell_label in np.unique(cells):
        # Skip if the cell label is 0 (background)
        if cell_label == 0:
            continue

        # Find the corresponding nucleus label for this cell
        nucleus_label = cell_label

        # Get the coordinates of the nucleus and cell regions
        nucleus_coords = np.argwhere(nuclei == nucleus_label)
        cell_coords = np.argwhere(cells == cell_label)

        # Update the cytoplasmic mask with the cell region
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label

        # Remove the nucleus region from the cytoplasmic mask
        cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)
