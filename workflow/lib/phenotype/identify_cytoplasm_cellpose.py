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
        ValueError: If the nuclei and cell masks are not reconciled (label sets differ).
    """
    # Each cell is paired with the same-label nucleus, so the masks must be reconciled.
    # Set equality, not count equality: the vectorized expression below is only
    # equivalent to the old per-label loop when the two label sets are identical.
    nuclei_labels = np.unique(nuclei)
    cells_labels = np.unique(cells)
    if set(nuclei_labels.tolist()) != set(cells_labels.tolist()):
        raise ValueError(
            f"Cannot identify cytoplasms: {len(nuclei_labels) - 1} nuclei vs "
            f"{len(cells_labels) - 1} cells, and their label sets differ. "
            "Cytoplasm needs reconciled masks; set `reconcile` "
            "(e.g. 'contained_in_cells' or 'consensus') in the config."
        )

    # Vectorized replacement for the former per-label Python loop.
    #
    # The old loop iterated every cell label L in ascending order and, for each,
    # wrote L over `cells == L` then zeroed `nuclei == L`. Because iterations ran
    # in ascending label order and each overwrote the previous, the net per-pixel
    # result was: keep the cell label C, UNLESS a nucleus label N (>0) with N >= C
    # covers the pixel. (Its own nucleus, N == C, zeros it within the same pass; a
    # larger-labelled nucleus, N > C, is processed later and also zeros it. A
    # smaller-labelled nucleus, N < C, is overwritten by the later cell-C pass and
    # therefore left as C — a label-order quirk, preserved here for output parity.)
    #
    # `(nuclei > 0) & (nuclei >= cells)` reproduces that mask exactly in a single
    # pass, eliminating the O(N_cells * N_pixels) `np.argwhere` loop. Bit-identity
    # against a verbatim copy of that loop is pinned in
    # tests/test_identify_cytoplasm_cellpose.py, including the N < C case that a
    # naive `np.where(nuclei > 0, 0, cells)` gets wrong.
    #
    # relies on nuclei/cells sharing the same label set (true for
    # matched cellpose masks, which reconcile_nuclei_cells relabels to a common
    # label set). This precondition is now ENFORCED by the set-equality gate
    # above — no longer a weak proxy: if a nucleus label ever existed that is
    # absent from `cells`, the old loop never zeroed it while this expression
    # would, so the gate raises first and this line never runs on such input.
    # The ascending-label semantics (keep C unless a nucleus N >= C covers the
    # pixel) hold only under that shared-label-set precondition.
    cytoplasms = np.where((nuclei > 0) & (nuclei >= cells), 0, cells)

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)
