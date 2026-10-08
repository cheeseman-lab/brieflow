"""Function for identifying and isolating the cytoplasm region."""

import numpy as np


def identify_cytoplasm_cellpose(nuclei, cells):
    """Identifies and isolates the cytoplasm region in an image based on the provided nuclei and cells masks.

    The result equals the former per-label loop, which wrote each cell label C in
    ascending order and then zeroed the same-label nucleus: a pixel is zeroed only
    when it carries a nucleus label N that is also a cell label, with N >= C.

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

    # Vectorized form of the old loop; bit-identity pinned in test_identify_cytoplasm_cellpose.py
    cytoplasms = np.where(
        (nuclei > 0) & (nuclei >= cells) & np.isin(nuclei, cells), 0, cells
    )

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)
