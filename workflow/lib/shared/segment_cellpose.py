"""Cellpose-based Image Segmentation!

This module provides functions for segmenting microscopy images using the Cellpose algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting cells and nuclei from various image types.
2. Image Preprocessing: Applying log scaling and other preprocessing techniques to images.
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships.
4. Mask Processing: Manipulating and refining segmentation masks.
5. Utility Functions: Supporting operations for image analysis and segmentation tasks.

COMPATIBILITY NOTE:
This module supports both Cellpose 3.x and 4.x with automatic version detection:

Cellpose 3.x (3.1.0):
- Supports models: cyto3, nuclei, cyto2
- Supports automatic diameter estimation
- CPSAM model NOT supported

Cellpose 4.x (4.0.4+):
- Supports ONLY cpsam model
- Standard models (cyto3, nuclei) NOT supported in 4.x
- Automatic diameter estimation NOT supported (specify diameters in config)
- Upgrade: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0

The code will automatically detect your Cellpose version and raise clear errors
for incompatible model/version combinations.

"""

import sys

import numpy as np
import pandas as pd

import cellpose
from cellpose.models import CellposeModel
from cellpose import models as cellpose_models
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border

from lib.shared.segmentation_utils import (
    image_log_scale,
    reconcile_nuclei_cells,
)

# Detect Cellpose version for compatibility checks
try:
    CELLPOSE_VERSION = tuple(map(int, cellpose.version.split(".")[:2]))
except (AttributeError, ValueError):
    # Fallback to safe default for backward compatibility
    CELLPOSE_VERSION = (3, 0)

CELLPOSE_4X = CELLPOSE_VERSION >= (4, 0)


def segment_cellpose(
    data,
    dapi_index,
    cyto_index,
    nuclei_diameter,
    cell_diameter,
    cellpose_model="cyto3",
    helper_index=None,
    cellpose_kwargs=dict(
        flow_threshold=0.4,
        cellprob_threshold=0,
        nuclei_flow_threshold=None,
        nuclei_cellprob_threshold=None,
        cell_flow_threshold=None,
        cell_cellprob_threshold=None,
    ),
    cells=True,
    reconcile="consensus",
    logscale=True,
    return_counts=False,
    gpu=False,
):
    """Segment cells using Cellpose algorithm.

    Args:
        data (numpy.ndarray): Multichannel image data.
        dapi_index (int): Index of DAPI channel.
        cyto_index (int): Index of cytoplasmic channel.
        nuclei_diameter (int): Estimated diameter of nuclei.
        cell_diameter (int): Estimated diameter of cells.
        cellpose_model (str, optional): Cellpose model type to use (e.g., 'cyto3', 'cpsam').
            Default is 'cyto3'. Use 'cpsam' for Cellpose-SAM (requires Cellpose 4.x).
        helper_index (int, optional): Index of helper channel for improved segmentation (CPSAM feature).
            Only used with multi-channel models. Default is None (blank channel).
        cellpose_kwargs (dict, optional): Additional keyword arguments for Cellpose, including:
            - flow_threshold (float): Default flow threshold for both nuclei and cells if specific ones not provided
            - cellprob_threshold (float): Default cell probability threshold for both nuclei and cells if specific ones not provided
            - nuclei_flow_threshold (float): Specific flow threshold for nuclei segmentation
            - nuclei_cellprob_threshold (float): Specific cell probability threshold for nuclei segmentation
            - cell_flow_threshold (float): Specific flow threshold for cell segmentation
            - cell_cellprob_threshold (float): Specific cell probability threshold for cell segmentation
        cells (bool, optional): Whether to segment both nuclei and cells or just nuclei.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        logscale (bool, optional): Whether to apply logarithmic transformation to image data.
        return_counts (bool, optional): Whether to return counts of nuclei and cells. Default is False.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.

    Returns:
        tuple or numpy.ndarray: If 'cells' is True, returns tuple of nuclei and cell segmentation masks,
        otherwise returns only nuclei segmentation mask. If return_counts is True, includes a dictionary of counts.
    """
    # Extract log_kwargs from cellpose_kwargs
    log_kwargs = cellpose_kwargs.pop("log_kwargs", dict())

    # Extract specific thresholds for nuclei and cells
    nuclei_flow_threshold = cellpose_kwargs.pop(
        "nuclei_flow_threshold", cellpose_kwargs.get("flow_threshold", 0.4)
    )
    nuclei_cellprob_threshold = cellpose_kwargs.pop(
        "nuclei_cellprob_threshold", cellpose_kwargs.get("cellprob_threshold", 0)
    )
    cell_flow_threshold = cellpose_kwargs.pop(
        "cell_flow_threshold", cellpose_kwargs.get("flow_threshold", 0.4)
    )
    cell_cellprob_threshold = cellpose_kwargs.pop(
        "cell_cellprob_threshold", cellpose_kwargs.get("cellprob_threshold", 0)
    )

    # Create separate kwargs dictionaries
    nuclei_kwargs = {
        "flow_threshold": nuclei_flow_threshold,
        "cellprob_threshold": nuclei_cellprob_threshold,
    }

    cell_kwargs = {
        "flow_threshold": cell_flow_threshold,
        "cellprob_threshold": cell_cellprob_threshold,
    }

    # Prepare data for Cellpose by creating a merged RGB image
    rgb = prepare_cellpose(
        data,
        dapi_index,
        cyto_index,
        helper_index=helper_index,
        logscale=logscale,
        log_kwargs=log_kwargs,
    )

    counts = {}

    # Perform cell segmentation using Cellpose
    if cells:
        if return_counts:
            nuclei, cells, seg_counts = segment_cellpose_rgb(
                rgb,
                nuclei_diameter,
                cell_diameter,
                cellpose_model=cellpose_model,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
                nuclei_kwargs=nuclei_kwargs,
                cell_kwargs=cell_kwargs,
            )
            counts.update(seg_counts)

        else:
            nuclei, cells = segment_cellpose_rgb(
                rgb,
                nuclei_diameter,
                cell_diameter,
                cellpose_model=cellpose_model,
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
        nuclei = segment_cellpose_nuclei_rgb(
            rgb,
            nuclei_diameter,
            cellpose_model=cellpose_model,
            gpu=gpu,
            **nuclei_kwargs,
        )
        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {counts['final_nuclei']}")
        counts_df = pd.DataFrame([counts])

        if return_counts:
            return nuclei, counts_df
        else:
            return nuclei


def prepare_cellpose(
    data, dapi_index, cyto_index, helper_index=None, logscale=True, log_kwargs=dict()
):
    """Prepare a three-channel RGB image for use with Cellpose.

    Args:
        data (list or numpy.ndarray): List or array containing DAPI and cytoplasmic channel images.
        dapi_index (int): Index of the DAPI channel in the data.
        cyto_index (int): Index of the cytoplasmic channel in the data.
        helper_index (int, optional): Index of helper channel for improved segmentation.
            If None, uses blank (zeros) channel. Default is None.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic and helper channels.
            Default is True.
        log_kwargs (dict, optional): Additional keyword arguments for log scaling.

    Returns:
        numpy.ndarray: Three-channel RGB image. Red is the helper channel (or blank if helper_index=None),
            green is the cytoplasmic channel, and blue is the DAPI channel.
    """
    # Extract DAPI and cytoplasmic channel images from the data
    dapi = data[dapi_index]
    cyto = data[cyto_index]

    # Extract helper channel if provided, otherwise use blank channel
    if helper_index is not None:
        helper = data[helper_index]
    else:
        helper = np.zeros_like(cyto)

    # Apply log scaling to the cytoplasmic and helper channels if specified
    if logscale:
        cyto = image_log_scale(cyto, **log_kwargs)
        # Safe normalization: check for zero max before division
        if cyto.max() > 0:
            cyto = cyto / cyto.max()

        # Apply log scaling to helper channel too
        helper = image_log_scale(helper, **log_kwargs)
        if helper.max() > 0:
            helper = helper / helper.max()

    # Normalize the intensity of the DAPI channel and scale it to the range [0, 1]
    dapi_upper = np.percentile(dapi, 99.5)
    # Safe normalization: check for zero before division
    if dapi_upper > 0:
        dapi = dapi / dapi_upper
    dapi = np.clip(dapi, 0, 1)

    # Convert the channels to uint8 format for RGB image creation
    red, green, blue = img_as_ubyte(helper), img_as_ubyte(cyto), img_as_ubyte(dapi)

    # Stack the channels to create the RGB image: [helper/red, cyto/green, dapi/blue]
    return np.array([red, green, blue])


def estimate_diameters(
    data,
    dapi_index,
    cyto_index,
    helper_index=None,
    channels=[2, 3],  # Default channels for cell estimation
    cellpose_model="cyto3",
    cellpose_kwargs=dict(flow_threshold=0.4, cellprob_threshold=0),
    gpu=False,
    logscale=True,
):
    """Estimate optimal cell diameter using Cellpose's diameter estimation.

    Note: Only supported with Cellpose 3.x. Cellpose 4.x does not have SizeModel.
          With Cellpose 4.x, you must specify nuclei_diameter and cell_diameter in config.

    Args:
        data (numpy.ndarray): Multichannel image data
        dapi_index (int): Index of DAPI channel
        cyto_index (int): Index of cytoplasmic channel
        helper_index (int, optional): Index of helper channel. Default is None.
        channels (list): Channel indices for diameter estimation [cytoplasm, nuclei]
        cellpose_model (str): Cellpose model type to use (e.g., 'cyto3', 'cpsam')
        cellpose_kwargs (dict): Additional keyword arguments for Cellpose
        gpu (bool): Whether to use GPU
        logscale (bool): Whether to apply log scaling to image

    Returns:
        tuple: Estimated diameters for (nuclei, cells)

    Raises:
        NotImplementedError: If using Cellpose 4.x (SizeModel not available)
    """
    # Check version compatibility
    if CELLPOSE_4X:
        raise NotImplementedError(
            "Automatic diameter estimation is not supported with Cellpose 4.x. "
            "SizeModel is not available in Cellpose 4.x. "
            "Please specify nuclei_diameter and cell_diameter explicitly in your config, "
            "or downgrade to Cellpose 3.x: uv pip install cellpose==3.1.0"
        )

    # Import SizeModel only for Cellpose 3.x (doesn't exist in 4.x)
    from cellpose.models import SizeModel

    # Prepare RGB image
    log_kwargs = cellpose_kwargs.pop(
        "log_kwargs", dict()
    )  # Extract log_kwargs from cellpose_kwargs
    rgb = prepare_cellpose(
        data,
        dapi_index,
        cyto_index,
        helper_index=helper_index,
        logscale=logscale,
        log_kwargs=log_kwargs,
    )

    # Find optimal nuclei diameter using explicit SizeModel
    print("Estimating nuclei diameters...")
    model_nuclei = CellposeModel(model_type="nuclei", gpu=gpu)
    size_model_nuclei = SizeModel(
        cp_model=model_nuclei, pretrained_size=cellpose_models.size_model_path("nuclei")
    )
    diam_nuclear, _ = size_model_nuclei.eval(rgb, channels=[3, 0])
    diam_nuclear = np.maximum(5.0, diam_nuclear)
    diam_nuclear = float(diam_nuclear)
    print(f"Estimated nuclear diameter: {diam_nuclear:.1f} pixels")

    # Find optimal cell diameter using explicit SizeModel
    print("Estimating cell diameters...")
    model_cyto = CellposeModel(model_type=cellpose_model, gpu=gpu)
    size_model_cyto = SizeModel(
        cp_model=model_cyto,
        pretrained_size=cellpose_models.size_model_path(cellpose_model),
    )
    diam_cell, _ = size_model_cyto.eval(rgb, channels=channels)
    diam_cell = np.maximum(5.0, diam_cell)
    diam_cell = float(diam_cell)
    print(f"Estimated cell diameter: {diam_cell:.1f} pixels")

    return diam_nuclear, diam_cell


def segment_cellpose_rgb(
    rgb,
    nuclei_diameter,
    cell_diameter,
    cellpose_model="cyto3",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
    gpu=False,
    nuclei_kwargs=None,
    cell_kwargs=None,
    **kwargs,
):
    """Segment nuclei and cells using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        cell_diameter (int): Diameter of cells for segmentation.
        cellpose_model (str, optional): Cellpose model type to use (e.g., 'cyto3', 'cpsam').
            Default is 'cyto3'. Cellpose 3.x: Use 'cyto3', 'nuclei', 'cyto2'.
            Cellpose 4.x: Use 'cpsam' only.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        remove_edges (bool, optional): Whether to remove nuclei and cells touching the image edges. Default is True.
        return_counts (bool, optional): Whether to return counts of nuclei and cells before reconciliation. Default is False.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        nuclei_kwargs (dict, optional): Specific parameters for nuclei segmentation. Default is None.
        cell_kwargs (dict, optional): Specific parameters for cell segmentation. Default is None.
        kwargs: Additional keyword arguments applied to both nuclei and cell segmentation if specific kwargs not provided.

    Returns:
        tuple: A tuple containing:
            - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of nuclei and cells at different stages if return_counts is True.

    Raises:
        ValueError: If model is incompatible with installed Cellpose version
    """
    # Validate model compatibility with Cellpose version
    if CELLPOSE_4X and cellpose_model != "cpsam":
        raise ValueError(
            f"Model '{cellpose_model}' requires Cellpose 3.x. "
            f"Cellpose 4.x only supports the 'cpsam' model. "
            f"Either change your config to use model='cpsam', "
            f"or downgrade Cellpose: uv pip install cellpose==3.1.0"
        )
    if not CELLPOSE_4X and cellpose_model == "cpsam":
        raise ValueError(
            f"CPSAM model requires Cellpose 4.x. "
            f"You have Cellpose {'.'.join(map(str, CELLPOSE_VERSION))}. "
            f"Upgrade with: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0"
        )

    # Create Cellpose models (different parameters for Cellpose 3.x vs 4.x)
    if CELLPOSE_4X:
        # Cellpose 4.x: Use pretrained_model parameter
        model_dapi = CellposeModel(pretrained_model="cpsam", gpu=gpu)
        model_cyto = CellposeModel(pretrained_model=cellpose_model, gpu=gpu)
    else:
        # Cellpose 3.x: Use model_type parameter
        model_dapi = CellposeModel(model_type="nuclei", gpu=gpu)
        model_cyto = CellposeModel(model_type=cellpose_model, gpu=gpu)

    # Set default kwargs if not provided
    if nuclei_kwargs is None:
        nuclei_kwargs = kwargs.copy()
    if cell_kwargs is None:
        cell_kwargs = kwargs.copy()

    counts = {}

    # Segment nuclei using nuclei-specific parameters
    # Pass only blue channel (DAPI) for nuclei segmentation
    nuclei, _, _ = model_dapi.eval(rgb[2], diameter=nuclei_diameter, **nuclei_kwargs)

    # Segment cells using cell-specific parameters
    # For CPSAM (Cellpose 4.x), use all 3 channels: [red/helper, green/cyto, blue/DAPI]
    # For standard models (Cellpose 3.x), use [green/cyto, blue/DAPI]
    if CELLPOSE_4X and cellpose_model == "cpsam":
        # CPSAM can use all 3 channels
        cells, _, _ = model_cyto.eval(
            rgb, diameter=cell_diameter, channels=[1, 2, 3], **cell_kwargs
        )
    else:
        # Standard cyto models use cytoplasm (green=index 2) and nuclei (blue=index 3) channels
        cells, _, _ = model_cyto.eval(
            rgb, diameter=cell_diameter, channels=[2, 3], **cell_kwargs
        )

    counts["initial_nuclei"] = (
        len(np.unique(nuclei)) - 1
    )  # Subtract 1 to exclude background
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['initial_nuclei']} nuclei before removing edges",
        file=sys.stderr,
    )
    print(
        f"found {counts['initial_cells']} cells before removing edges", file=sys.stderr
    )

    # Remove nuclei and cells touching the image edges if specified
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

    # Reconcile nuclei and cells if specified
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


def segment_cellpose_nuclei_rgb(
    rgb,
    nuclei_diameter,
    cellpose_model="nuclei",
    gpu=False,
    remove_edges=True,
    **kwargs,
):
    """Segment nuclei using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        cellpose_model (str, optional): Cellpose model to use. Default is "nuclei".
            Can also use "cyto3" or other models if nuclei-specific model produces poor results.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        remove_edges (bool, optional): Whether to remove nuclei touching the image edges. Default is True.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Labeled segmentation mask of nuclei.
    """
    # Validate model compatibility with Cellpose version
    if CELLPOSE_4X and cellpose_model != "cpsam":
        raise ValueError(
            f"Model '{cellpose_model}' requires Cellpose 3.x. "
            f"Cellpose 4.x only supports the 'cpsam' model. "
            f"Either change your config to use model='cpsam', "
            f"or downgrade Cellpose: uv pip install cellpose==3.1.0"
        )
    if not CELLPOSE_4X and cellpose_model == "cpsam":
        raise ValueError(
            f"CPSAM model requires Cellpose 4.x. "
            f"You have Cellpose {'.'.join(map(str, CELLPOSE_VERSION))}. "
            f"Upgrade with: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0"
        )

    # Create Cellpose model (different parameters for Cellpose 3.x vs 4.x)
    if CELLPOSE_4X:
        # Cellpose 4.x: Use pretrained_model parameter
        model = CellposeModel(pretrained_model=cellpose_model, gpu=gpu)
    else:
        # Cellpose 3.x: Use model_type parameter
        model = CellposeModel(model_type=cellpose_model, gpu=gpu)

    # Segment nuclei using CellposeModel from the RGB image
    # Pass only blue channel (DAPI) for nuclei segmentation
    nuclei, _, _ = model.eval(rgb[2], diameter=nuclei_diameter, **kwargs)

    # Print the number of nuclei found before and after removing edges
    print(
        f"found {len(np.unique(nuclei))} nuclei before removing edges", file=sys.stderr
    )

    # Remove nuclei touching the image edges if specified
    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)

    # Print the final number of nuclei after processing
    print(f"found {len(np.unique(nuclei))} final nuclei", file=sys.stderr)

    # Return the segmented nuclei
    return nuclei
