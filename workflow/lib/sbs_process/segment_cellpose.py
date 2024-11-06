"""Cellpose-based Image Segmentation!

This module provides functions for segmenting microscopy images using the Cellpose algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting cells and nuclei from various image types.
2. Image Preprocessing: Applying log scaling and other preprocessing techniques to images.
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships.
4. Mask Processing: Manipulating and refining segmentation masks.
5. Utility Functions: Supporting operations for image analysis and segmentation tasks.

"""

import sys

import numpy as np
import pandas as pd

from cellpose.models import Cellpose
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
from skimage.segmentation import clear_border


def segment_cellpose(
    data,
    dapi_index,
    cyto_index,
    nuclei_diameter,
    cell_diameter,
    cyto_model="cyto",
    cellpose_kwargs=dict(),
    cells=True,
    reconcile="consensus",
    logscale=True,
    return_counts=False,
):
    """Segment cells using Cellpose algorithm.

    Args:
        data (numpy.ndarray): Multichannel image data.
        dapi_index (int): Index of DAPI channel.
        cyto_index (int): Index of cytoplasmic channel.
        nuclei_diameter (int): Estimated diameter of nuclei.
        cell_diameter (int): Estimated diameter of cells.
        cyto_model (str, optional): Type of cytoplasmic model to use. Default is 'cyto'.
        logscale (bool, optional): Whether to apply logarithmic transformation to image data.
        cellpose_kwargs (dict, optional): Additional keyword arguments for Cellpose.
        cells (bool, optional): Whether to segment both nuclei and cells or just nuclei.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        return_counts (bool, optional): Whether to return counts of nuclei and cells. Default is False.

    Returns:
        tuple or numpy.ndarray: If 'cells' is True, returns tuple of nuclei and cell segmentation masks,
        otherwise returns only nuclei segmentation mask. If return_counts is True, includes a dictionary of counts.
    """
    # Prepare data for Cellpose by creating a merged RGB image
    log_kwargs = cellpose_kwargs.pop(
        "log_kwargs", dict()
    )  # Extract log_kwargs from cellpose_kwargs
    rgb = prepare_cellpose(
        data, dapi_index, cyto_index, logscale, log_kwargs=log_kwargs
    )

    counts = {}

    # Perform cell segmentation using Cellpose
    if cells:
        if return_counts:
            nuclei, cells, seg_counts = segment_cellpose_rgb(
                rgb,
                nuclei_diameter,
                cell_diameter,
                cyto_model=cyto_model,
                reconcile=reconcile,
                return_counts=True,
                **cellpose_kwargs,
            )
            counts.update(seg_counts)

        else:
            nuclei, cells = segment_cellpose_rgb(
                rgb,
                nuclei_diameter,
                cell_diameter,
                reconcile=reconcile,
                **cellpose_kwargs,
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
        nuclei = segment_cellpose_nuclei_rgb(rgb, nuclei_diameter, **cellpose_kwargs)
        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {counts['final_nuclei']}")
        counts_df = pd.DataFrame([counts])

        if return_counts:
            return nuclei, counts_df
        else:
            return nuclei


def prepare_cellpose(data, dapi_index, cyto_index, logscale=True, log_kwargs=dict()):
    """Prepare a three-channel RGB image for use with the Cellpose GUI.

    Args:
        data (list or numpy.ndarray): List or array containing DAPI and cytoplasmic channel images.
        dapi_index (int): Index of the DAPI channel in the data.
        cyto_index (int): Index of the cytoplasmic channel in the data.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic channel. Default is True.
        log_kwargs (dict, optional): Additional keyword arguments for log scaling.

    Returns:
        numpy.ndarray: Three-channel RGB image prepared for use with Cellpose GUI.
    """
    # Extract DAPI and cytoplasmic channel images from the data
    dapi = data[dapi_index]
    cyto = data[cyto_index]

    # Create a blank array with the same shape as the DAPI channel
    blank = np.zeros_like(dapi)

    # Apply log scaling to the cytoplasmic channel if specified
    if logscale:
        cyto = image_log_scale(cyto, **log_kwargs)
        cyto /= cyto.max()  # Normalize the image for uint8 conversion

    # Normalize the intensity of the DAPI channel and scale it to the range [0, 1]
    dapi_upper = np.percentile(dapi, 99.5)
    dapi = dapi / dapi_upper
    dapi[dapi > 1] = 1

    # Convert the channels to uint8 format for RGB image creation
    red, green, blue = img_as_ubyte(blank), img_as_ubyte(cyto), img_as_ubyte(dapi)

    # Stack the channels to create the RGB image and transpose the dimensions
    # return np.array([red, green, blue]).transpose([1, 2, 0])
    return np.array([red, green, blue])


def segment_cellpose_rgb(
    rgb,
    nuclei_diameter,
    cell_diameter,
    gpu=False,
    cyto_model="cyto",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
):
    """Segment nuclei and cells using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        cell_diameter (int): Diameter of cells for segmentation.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        cyto_model (str, optional): Type of cytoplasmic model to use. Default is 'cyto'.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is 'consensus'.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic channel. Default is True.
        remove_edges (bool, optional): Whether to remove nuclei and cells touching the image edges. Default is True.
        return_counts (bool, optional): Whether to return counts of nuclei and cells before reconciliation. Default is False.

    Returns:
        tuple: A tuple containing:
            - nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of nuclei and cells at different stages if return_counts is True.
    """
    # Instantiate Cellpose models for nuclei and cytoplasmic segmentation
    model_dapi = Cellpose(model_type="nuclei", gpu=gpu)
    model_cyto = Cellpose(model_type=cyto_model, gpu=gpu)

    counts = {}

    # Segment nuclei and cells using Cellpose from the RGB image
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)
    cells, _, _, _ = model_cyto.eval(rgb, channels=[2, 3], diameter=cell_diameter)

    counts["initial_nuclei"] = (
        len(np.unique(nuclei)) - 1
    )  # Subtract 1 to exclude background
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f'found {counts["initial_nuclei"]} nuclei before removing edges',
        file=sys.stderr,
    )
    print(
        f'found {counts["initial_cells"]} cells before removing edges', file=sys.stderr
    )

    # Remove nuclei and cells touching the image edges if specified
    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)
        cells = clear_border(cells)

    counts["after_edge_removal_nuclei"] = len(np.unique(nuclei)) - 1
    counts["after_edge_removal_cells"] = len(np.unique(cells)) - 1

    print(
        f'found {counts["after_edge_removal_nuclei"]} nuclei before reconciling',
        file=sys.stderr,
    )
    print(
        f'found {counts["after_edge_removal_cells"]} cells before reconciling',
        file=sys.stderr,
    )

    # Reconcile nuclei and cells if specified
    if reconcile:
        print(f"reconciling masks with method how={reconcile}")
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)

    counts["final_cells"] = len(np.unique(cells)) - 1
    print(
        f'found {counts["final_cells"]} nuclei/cells after reconciling', file=sys.stderr
    )

    if return_counts:
        return nuclei, cells, counts
    else:
        return nuclei, cells


def segment_cellpose_nuclei_rgb(
    rgb, nuclei_diameter, gpu=False, remove_edges=True, **kwargs
):
    """Segment nuclei using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        nuclei_diameter (int): Diameter of nuclei for segmentation.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        remove_edges (bool, optional): Whether to remove nuclei touching the image edges. Default is True.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Labeled segmentation mask of nuclei.
    """
    # Instantiate Cellpose model for nuclei segmentation
    model_dapi = Cellpose(model_type="nuclei", gpu=gpu)

    # Segment nuclei using Cellpose from the RGB image
    nuclei, _, _, _ = model_dapi.eval(rgb, channels=[3, 0], diameter=nuclei_diameter)

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


def reconcile_nuclei_cells(nuclei, cells, how="consensus"):
    """Reconcile nuclei and cells labels based on their overlap.

    Args:
        nuclei (numpy.ndarray): Nuclei mask.
        cells (numpy.ndarray): Cell mask.
        how (str, optional): Method to reconcile labels.
            - 'consensus': Only keep nucleus-cell pairs where label matches are unique.
            - 'contained_in_cells': Keep multiple nuclei for a single cell but merge them.

    Returns:
        tuple: Tuple containing the reconciled nuclei and cells masks.
    """
    from skimage.morphology import erosion

    def get_unique_label_map(regions, keep_multiple=False):
        """Get unique label map from regions.

        Args:
            regions (list): List of regions.
            keep_multiple (bool, optional): Whether to keep multiple labels for each region.

        Returns:
            dict: Dictionary containing the label map.
        """
        label_map = {}
        for region in regions:
            intensity_image = region.intensity_image[region.intensity_image > 0]
            labels = np.unique(intensity_image)
            if keep_multiple:
                label_map[region.label] = labels
            elif len(labels) == 1:
                label_map[region.label] = labels[0]
        return label_map

    # Erode nuclei to prevent overlapping with cells
    nuclei_eroded = center_pixels(nuclei)

    # Get unique label maps for nuclei and cells
    nucleus_map = get_unique_label_map(
        regionprops(nuclei_eroded, intensity_image=cells)
    )
    if how == "contained_in_cells":
        cell_map = get_unique_label_map(
            regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True
        )
    else:
        cell_map = get_unique_label_map(
            regionprops(cells, intensity_image=nuclei_eroded)
        )

    # Keep only nucleus-cell pairs with matching labels
    keep = []
    for nucleus in nucleus_map:
        try:
            if how == "contained_in_cells":
                if nucleus in cell_map[nucleus_map[nucleus]]:
                    keep.append([nucleus, nucleus_map[nucleus]])
            else:
                if cell_map[nucleus_map[nucleus]] == nucleus:
                    keep.append([nucleus, nucleus_map[nucleus]])
        except KeyError:
            pass

    # If no matches found, return zero arrays
    if len(keep) == 0:
        return np.zeros_like(nuclei), np.zeros_like(cells)

    # Extract nuclei and cells to keep
    keep_nuclei, keep_cells = zip(*keep)

    # Reassign labels based on the reconciliation method
    if how == "contained_in_cells":
        nuclei = relabel_array(
            nuclei, {nuclei_label: cell_label for nuclei_label, cell_label in keep}
        )
        cells[~np.isin(cells, keep_cells)] = 0
        labels, cell_indices = np.unique(cells, return_inverse=True)
        _, nuclei_indices = np.unique(nuclei, return_inverse=True)
        cells = np.arange(0, labels.shape[0])[cell_indices.reshape(*cells.shape)]
        nuclei = np.arange(0, labels.shape[0])[nuclei_indices.reshape(*nuclei.shape)]
    else:
        nuclei = relabel_array(
            nuclei, {label: i + 1 for i, label in enumerate(keep_nuclei)}
        )
        cells = relabel_array(
            cells, {label: i + 1 for i, label in enumerate(keep_cells)}
        )

    # Convert arrays to integers
    nuclei, cells = nuclei.astype(int), cells.astype(int)
    return nuclei, cells


def center_pixels(label_image):
    """Assign labels to center pixels of regions in a labeled image.

    Args:
        label_image (numpy.ndarray): Labeled image.

    Returns:
        numpy.ndarray: Image with labels assigned to center pixels of regions.
    """
    ultimate = np.zeros_like(label_image)  # Initialize an array to store the result
    for r in regionprops(label_image):  # Iterate over regions in the labeled image
        # Calculate the mean coordinates of the bounding box of the region
        i, j = np.array(r.bbox).reshape(2, 2).mean(axis=0).astype(int)
        # Assign the label of the region to the center pixel
        ultimate[i, j] = r.label
    return ultimate  # Return the image with labels assigned to center pixels


def image_log_scale(data, bottom_percentile=10, floor_threshold=50, ignore_zero=True):
    """Apply log scaling to an image.

    Args:
        data (numpy.ndarray): Input image data.
        bottom_percentile (int, optional): Percentile value for determining the bottom threshold. Default is 10.
        floor_threshold (int, optional): Floor threshold for cutting out noisy bits. Default is 50.
        ignore_zero (bool, optional): Whether to ignore zero values in the data. Default is True.

    Returns:
        numpy.ndarray: Scaled image data after log scaling.
    """
    import numpy as np

    # Convert input data to float
    data = data.astype(float)

    # Select data based on whether to ignore zero values or not
    if ignore_zero:
        data_perc = data[data > 0]
    else:
        data_perc = data

    # Determine the bottom percentile value
    bottom = np.percentile(data_perc, bottom_percentile)

    # Set values below the bottom percentile to the bottom value
    data[data < bottom] = bottom

    # Apply log scaling with floor threshold
    scaled = np.log10(data - bottom + 1)

    # Cut out noisy bits based on the floor threshold
    floor = np.log10(floor_threshold)
    scaled[scaled < floor] = floor

    # Subtract the floor value
    return scaled - floor


def relabel_array(arr, new_label_dict):
    """Map values in an integer array based on `new_label_dict`, a dictionary from old to new values.

    Args:
        arr (numpy.ndarray): The input integer array to be relabeled.
        new_label_dict (dict): A dictionary mapping old values to new values.

    Returns:
        numpy.ndarray: The relabeled integer array.

    Notes:
    - The function iterates through the items in `new_label_dict` and maps old values to new values in the array.
    - Values in the array that do not have a corresponding mapping in `new_label_dict` remain unchanged.
    """
    n = arr.max()  # Find the maximum value in the array
    arr_ = np.zeros(n + 1)  # Initialize an array to store the relabeled values
    for old_val, new_val in new_label_dict.items():
        if old_val <= n:  # Check if the old value is within the range of the array
            arr_[old_val] = (
                new_val  # Map the old value to the new value in the relabeling array
            )
    return arr_[arr]  # Return the relabeled array
