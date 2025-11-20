"""StarDist-based Image Segmentation!

This module provides functions for segmenting microscopy images using the StarDist algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting both cellular components using StarDist
2. Image Preprocessing: Applying intensity normalization and preprocessing techniques
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships
4. Mask Processing: Manipulating and refining segmentation masks
5. Utility Functions: Supporting operations for image analysis and segmentation tasks
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.segmentation import clear_border

from lib.shared.segmentation_utils import reconcile_nuclei_cells


def segment_stardist(
    data,
    dapi_index,
    cyto_index,
    model_type="2D_versatile_fluo",
    stardist_kwargs=dict(
        prob_threshold=0.479071,
        nms_threshold=0.3,
        nuclei_prob_threshold=None,
        nuclei_nms_threshold=None,
        cell_prob_threshold=None,
        cell_nms_threshold=None,
    ),
    cells=True,
    reconcile="consensus",
    return_counts=False,
    gpu=False,
):
    """Segment cells using StarDist algorithm with separate parameters for nuclei and cells.

    Args:
        data: Multichannel image data
        dapi_index: Index of DAPI channel
        cyto_index: Index of cytoplasmic channel
        model_type: StarDist model type to use
        stardist_kwargs: Additional keyword arguments for StarDist, including:
            - prob_threshold: Default probability threshold for both nuclei and cells
            - nms_threshold: Default NMS threshold for both nuclei and cells
            - nuclei_prob_threshold: Specific probability threshold for nuclei segmentation
            - nuclei_nms_threshold: Specific NMS threshold for nuclei segmentation
            - cell_prob_threshold: Specific probability threshold for cell segmentation
            - cell_nms_threshold: Specific NMS threshold for cell segmentation
        cells: Whether to segment both nuclei and cells or just nuclei
        reconcile: Method for reconciling nuclei and cells
        return_counts: Whether to return counts of nuclei and cells
        gpu: Whether to use GPU for segmentation

    Returns:
        Segmentation masks with optional counts
    """
    # Extract specific thresholds for nuclei and cells
    nuclei_prob_threshold = stardist_kwargs.pop(
        "nuclei_prob_threshold", stardist_kwargs.get("prob_threshold", 0.479071)
    )
    nuclei_nms_threshold = stardist_kwargs.pop(
        "nuclei_nms_threshold", stardist_kwargs.get("nms_threshold", 0.3)
    )
    cell_prob_threshold = stardist_kwargs.pop(
        "cell_prob_threshold", stardist_kwargs.get("prob_threshold", 0.479071)
    )
    cell_nms_threshold = stardist_kwargs.pop(
        "cell_nms_threshold", stardist_kwargs.get("nms_threshold", 0.3)
    )

    # Create separate kwargs dictionaries
    nuclei_kwargs = {
        "prob_thresh": nuclei_prob_threshold,
        "nms_thresh": nuclei_nms_threshold,
    }
    cell_kwargs = {
        "prob_thresh": cell_prob_threshold,
        "nms_thresh": cell_nms_threshold,
    }

    # Prepare channels for StarDist
    dapi = prepare_channel(data[dapi_index])
    cyto = prepare_channel(data[cyto_index])

    counts = {}

    # Perform cell segmentation using StarDist
    if cells:
        if return_counts:
            nuclei, cells, seg_counts = segment_stardist_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
                nuclei_kwargs=nuclei_kwargs,
                cell_kwargs=cell_kwargs,
            )
            counts.update(seg_counts)

        else:
            nuclei, cells = segment_stardist_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                gpu=gpu,
                nuclei_kwargs=nuclei_kwargs,
                cell_kwargs=cell_kwargs,
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
        nuclei = segment_stardist_nuclei(
            dapi, model_type=model_type, gpu=gpu, **nuclei_kwargs
        )

        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {counts['final_nuclei']}")

        counts_df = pd.DataFrame([counts])

        if return_counts:
            return nuclei, counts_df
        else:
            return nuclei


def prepare_channel(data):
    """Prepare channel data for segmentation using StarDist's normalization.

    Args:
        data: Input channel data

    Returns:
        Processed channel data
    """
    # Use StarDist's recommended normalization
    return normalize(data, 1, 99.8, axis=None)


def segment_stardist_multichannel(
    dapi,
    cyto,
    model_type="2D_versatile_fluo",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
    gpu=False,
    nuclei_kwargs=None,
    cell_kwargs=None,
    **kwargs,
):
    """Segment nuclei and cells using the StarDist algorithm with separate parameters.

    Args:
        dapi: DAPI channel data
        cyto: Cytoplasmic channel data
        model_type: StarDist model type to use
        reconcile: Method for reconciling nuclei and cells
        remove_edges: Whether to remove edges from the masks
        return_counts: Whether to return counts of nuclei and cells
        gpu: Whether to use GPU for segmentation
        nuclei_kwargs: Specific parameters for nuclei segmentation
        cell_kwargs: Specific parameters for cell segmentation
        kwargs: Additional keyword arguments applied to both if specific kwargs not provided

    Returns:
        tuple: A tuple containing:
            - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of nuclei and cells at different stages if return_counts is True.
    """
    # Initialize StarDist models for nuclei and cytoplasmic segmentation
    model_nuclei = StarDist2D.from_pretrained(model_type)
    model_cells = StarDist2D.from_pretrained(model_type)

    # Set default kwargs if not provided
    if nuclei_kwargs is None:
        nuclei_kwargs = kwargs.copy()
    if cell_kwargs is None:
        cell_kwargs = kwargs.copy()

    counts = {}

    if gpu:
        model_nuclei.config.use_gpu = True
        model_cells.config.use_gpu = True

    # Segment nuclei using nuclei-specific parameters
    nuclei, _ = model_nuclei.predict_instances(dapi, **nuclei_kwargs)

    # Segment cells using cell-specific parameters
    cells, _ = model_cells.predict_instances(cyto, **cell_kwargs)

    counts["initial_nuclei"] = len(np.unique(nuclei)) - 1
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['initial_nuclei']} nuclei before removing edges",
        file=sys.stderr,
    )
    print(
        f"found {counts['initial_cells']} cells before removing edges", file=sys.stderr
    )

    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)

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

    if reconcile:
        print(f"reconciling masks with method how={reconcile}")
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)

    counts["final_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['final_cells']} nuclei/cells after reconciling", file=sys.stderr
    )

    if return_counts:
        return nuclei, cells, counts
    else:
        return nuclei, cells


def segment_stardist_nuclei(
    dapi,
    model_type="2D_versatile_fluo",
    gpu=False,
    remove_edges=True,
    **kwargs,
):
    """Segment nuclei using the StarDist algorithm.

    Args:
        dapi: DAPI channel data
        model_type: StarDist model type to use
        remove_edges: Whether to remove edges from the masks
        gpu: Whether to use GPU for segmentation
        **kwargs: Parameters for StarDist segmentation including:
                 - prob_thresh: Probability threshold for segmentation
                 - nms_thresh: Non-maximum suppression threshold for segmentation
    Returns:
        Segmented nuclei masks
    """
    # Initialize StarDist model
    model = StarDist2D.from_pretrained(model_type)
    if gpu:
        model.config.use_gpu = True

    # Segment nuclei with specified parameters
    nuclei, _ = model.predict_instances(dapi, **kwargs)

    print(
        f"found {len(np.unique(nuclei))} nuclei before removing edges", file=sys.stderr
    )

    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)

    print(f"found {len(np.unique(nuclei))} final nuclei", file=sys.stderr)

    return nuclei
