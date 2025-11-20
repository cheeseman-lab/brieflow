"""MicroSAM-based Image Segmentation!

This module provides functions for segmenting microscopy images using the MicroSAM algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting both cellular components using SAM
2. Image Preprocessing: Applying intensity normalization and preprocessing techniques
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships
4. Mask Processing: Manipulating and refining segmentation masks
5. Utility Functions: Supporting operations for image analysis and segmentation tasks
"""

import sys
import numpy as np
import pandas as pd

from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation,
)
from skimage.segmentation import clear_border

from lib.shared.segmentation_utils import reconcile_nuclei_cells


def segment_microsam(
    data,
    dapi_index,
    cyto_index,
    model_type="vit_b_lm",
    cells=True,
    reconcile="consensus",
    return_counts=False,
    gpu=False,
):
    """Segment cells using MicroSAM algorithm.

    Args:
        data (numpy.ndarray): Multichannel image data.
        dapi_index (int): Index of DAPI channel.
        cyto_index (int): Index of cytoplasmic channel.
        model_type (str, optional): MicroSAM model type to use. Default is 'vit_b_lm'.
        microsam_kwargs (dict, optional): Additional parameters for MicroSAM segmentation.
        cells (bool, optional): Whether to segment both nuclei and cells or just nuclei. Default is True.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        return_counts (bool, optional): Whether to return counts of nuclei and cells. Default is False.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.

    Returns:
        tuple or numpy.ndarray: If 'cells' is True, returns tuple of nuclei and cell segmentation masks,
        otherwise returns only nuclei segmentation mask. If return_counts is True, includes a dictionary of counts.
    """
    # Prepare channels for MicroSAM
    dapi = data[dapi_index]
    cyto = data[cyto_index]
    counts = {}

    # Perform cell segmentation
    if cells:
        if return_counts:
            nuclei, cells, seg_counts = segment_microsam_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
            )
            counts.update(seg_counts)
        else:
            nuclei, cells = segment_microsam_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                gpu=gpu,
            )

        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        counts["final_cells"] = len(np.unique(cells)) - 1
        counts_df = pd.DataFrame([counts])

        print(f"Number of nuclei segmented: {counts['final_nuclei']}")
        print(f"Number of cells segmented: {counts['final_cells']}")

        if return_counts:
            return nuclei, cells, counts_df
        else:
            return nuclei, cells
    else:
        nuclei = segment_microsam_nuclei(dapi, model_type=model_type)

        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {counts['final_nuclei']}")

        counts_df = pd.DataFrame([counts])

        if return_counts:
            return nuclei, counts_df
        else:
            return nuclei


def segment_microsam_multichannel(
    dapi,
    cyto,
    model_type="vit_b_lm",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
    gpu=False,
):
    """Segment nuclei and cells using the MicroSAM algorithm with InstanceSegmentationWithDecoder.

    Args:
        dapi (numpy.ndarray): DAPI channel image
        cyto (numpy.ndarray): Cytoplasmic channel image
        model_type (str, optional): MicroSAM model type to use
        reconcile (str, optional): Method for reconciling nuclei and cells
        remove_edges (bool, optional): Whether to remove nuclei and cells touching image edges
        return_counts (bool, optional): Whether to return counts of nuclei and cells
        gpu (bool, optional): Whether to use GPU for segmentation

    Returns:
        Segmentation masks with optional counts
    """
    counts = {}

    # Step 1: Initialize the model and decoder
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type, checkpoint_path=None
    )

    # Step 2: Computation of image embeddings
    dapi_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=dapi, ndim=2
    )
    cyto_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=cyto, ndim=2
    )

    # Step 3: Nuclei segmentation with decoder
    ais_nuclei = InstanceSegmentationWithDecoder(predictor, decoder)
    ais_nuclei.initialize(image=dapi, image_embeddings=dapi_embeddings)
    nuclei_prediction = ais_nuclei.generate()

    # Step 4: Cell segmentation with decoder
    ais_cells = InstanceSegmentationWithDecoder(predictor, decoder)
    ais_cells.initialize(image=cyto, image_embeddings=cyto_embeddings)
    cells_prediction = ais_cells.generate()

    # Check if we got any predictions and convert mask data to segmentation
    if nuclei_prediction:
        nuclei = mask_data_to_segmentation(nuclei_prediction, with_background=True)
    else:
        print("Warning: No nuclei detected in the DAPI channel", file=sys.stderr)
        nuclei = np.zeros_like(dapi, dtype="uint32")

    if cells_prediction:
        cells = mask_data_to_segmentation(cells_prediction, with_background=True)
    else:
        print("Warning: No cells detected in the cytoplasmic channel", file=sys.stderr)
        cells = np.zeros_like(cyto, dtype="uint32")

    # Track initial counts
    counts["initial_nuclei"] = len(np.unique(nuclei)) - 1
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['initial_nuclei']} nuclei before removing edges",
        file=sys.stderr,
    )
    print(
        f"found {counts['initial_cells']} cells before removing edges", file=sys.stderr
    )

    # Remove objects touching edges if specified
    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)

    # Update counts after edge removal
    counts["after_edge_removal_nuclei"] = len(np.unique(nuclei)) - 1
    counts["after_edge_removal_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['after_edge_removal_nuclei']} nuclei before reconciling",
        file=sys.stderr,
    )
    print(
        f"found {counts['after_edge_removal_cells']} cells before reconciling",
        file=sys.stderr,
    )

    # Reconcile nuclei and cells if specified
    if (
        reconcile
        and counts["after_edge_removal_nuclei"] > 0
        and counts["after_edge_removal_cells"] > 0
    ):
        print(f"reconciling masks with method how={reconcile}")
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)

    # Final count after reconciliation
    counts["final_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['final_cells']} nuclei/cells after reconciling", file=sys.stderr
    )

    if return_counts:
        return nuclei, cells, counts
    else:
        return nuclei, cells


def segment_microsam_nuclei(
    dapi,
    model_type="vit_b_lm",
    remove_edges=True,
    gpu=False,
):
    """Segment nuclei using the MicroSAM algorithm with InstanceSegmentationWithDecoder.

    Args:
        dapi (numpy.ndarray): DAPI channel image
        model_type (str, optional): MicroSAM model type to use
        remove_edges (bool, optional): Whether to remove nuclei touching the image edges
        gpu (bool, optional): Whether to use GPU for segmentation

    Returns:
        Labeled segmentation mask of nuclei
    """
    # Step 1: Initialize the model and decoder
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type, checkpoint_path=None
    )

    # Step 2: Compute image embeddings for DAPI channel
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=dapi, ndim=2
    )

    # Step 3: Create instance segmentation with decoder
    ais_nuclei = InstanceSegmentationWithDecoder(predictor, decoder)

    # Step 4: Initialize with precomputed embeddings
    ais_nuclei.initialize(image=dapi, image_embeddings=image_embeddings)

    # Step 5: Generate segmentation
    nuclei_masks = ais_nuclei.generate()

    # Check if we got any predictions
    if nuclei_masks:
        nuclei = mask_data_to_segmentation(nuclei_masks, with_background=True)
    else:
        print("Warning: No nuclei detected in the DAPI channel", file=sys.stderr)
        nuclei = np.zeros_like(dapi, dtype="uint32")

    # Print the number of nuclei found before and after removing edges
    nuclei_count = len(np.unique(nuclei)) - 1
    print(f"found {nuclei_count} nuclei before removing edges", file=sys.stderr)

    # Remove nuclei touching the image edges if specified
    if remove_edges and nuclei_count > 0:
        print("removing edges")
        nuclei = clear_border(nuclei)

    # Print the final number of nuclei after processing
    final_count = len(np.unique(nuclei)) - 1
    print(f"found {final_count} final nuclei", file=sys.stderr)

    return nuclei
