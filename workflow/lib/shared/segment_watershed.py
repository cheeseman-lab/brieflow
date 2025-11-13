"""Watershed-based Image Segmentation!

This module provides functions for segmenting microscopy images using the Watershed algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Nuclei Segmentation: Segmenting cells and nuclei from various image types.
2. Image Preprocessing: Applying filtering and preprocessing techniques to images.
3. Label Reconciliation: Reconciling nuclei and cell labels based on their spatial relationships.
4. Mask Processing: Manipulating and refining segmentation masks.
5. Utility Functions: Supporting operations for image analysis and segmentation tasks.

"""

import sys
import warnings
import numpy as np
import pandas as pd

import skimage
from skimage.measure import label
from skimage.segmentation import clear_border, watershed, relabel_sequential
from skimage.morphology import (
    disk,
    binary_erosion,
    binary_dilation,
    remove_small_objects,
)
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, gaussian, rank

from lib.shared.segmentation_utils import reconcile_nuclei_cells
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte


def segment_watershed(
    data,
    nuclei_threshold,
    nuclei_area_min,
    nuclei_area_max,
    cell_threshold,
    cells=True,
    smooth=1.35,
    radius=15,
    return_counts=False,
    reconcile=None,
):
    """Segment cells using watershed method.

    Args:
        data (numpy.ndarray): Image data for segmentation.
        nuclei_threshold (float): Threshold for nuclei segmentation.
        nuclei_area_min (float): Minimum area for retaining nuclei after segmentation.
        nuclei_area_max (float): Maximum area for retaining nuclei after segmentation.
        cell_threshold (float): Threshold used for cell boundary segmentation.
        cells (bool, optional): Whether to segment both nuclei and cells or just nuclei. Default is True.
        smooth (float, optional): Size of Gaussian kernel for smoothing. Default is 1.35.
        radius (float, optional): Radius of disk for local thresholding. Default is 15.
        return_counts (bool, optional): Whether to return counts of nuclei and cells. Default is False.
        reconcile (str, optional): Method for reconciling nuclei and cells. Default is None.

    Returns:
        tuple or numpy.ndarray: If 'cells' is True, returns tuple of nuclei and cell segmentation masks,
        otherwise returns only nuclei segmentation mask. If return_counts is True, includes a dictionary of counts.
    """
    # If SBS data, image will have 4 dimensions
    if data.ndim == 4:
        # Select first cycle
        nuclei_data = data[0]
    elif data.ndim == 3:
        nuclei_data = data
    else:
        nuclei_data = data

    # Segment nuclei using the segment_nuclei method
    nuclei = segment_nuclei(
        nuclei_data,
        nuclei_threshold,
        nuclei_area_min,
        nuclei_area_max,
        smooth=smooth,
        radius=radius,
    )

    counts = {}
    counts["nuclei"] = len(np.unique(nuclei)) - 1  # Subtract 1 to exclude background

    if not cells:
        if return_counts:
            counts_df = pd.DataFrame([counts])
            return nuclei, counts_df
        else:
            return nuclei

    # Segment cells using the segment_cells method
    cells = segment_cells(data, nuclei, cell_threshold)

    counts["cells"] = len(np.unique(cells)) - 1  # Subtract 1 to exclude background

    # Reconcile nuclei and cells if specified
    if reconcile:
        print(f"reconciling masks with method how={reconcile}")
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
        counts["reconciled_nuclei"] = len(np.unique(nuclei)) - 1
        counts["reconciled_cells"] = len(np.unique(cells)) - 1

    if return_counts:
        counts_df = pd.DataFrame([counts])
        return nuclei, cells, counts_df
    else:
        return nuclei, cells


def segment_nuclei(data, threshold, area_min, area_max, smooth=1.35, radius=15):
    """Find nuclei from DAPI channel.

    Uses local mean filtering to find cell foreground from aligned but unfiltered data,
    then filters identified regions by mean intensity threshold and area ranges.

    Args:
        data (numpy.ndarray or list): Image data.
            If numpy.ndarray, expected dimensions are (CHANNEL, I, J) with the DAPI channel in channel index 0.
            If list, the first element is assumed to be the DAPI channel.
            Can also be a single-channel DAPI image of dimensions (I, J).
        threshold (float): Foreground regions with mean DAPI intensity greater than `threshold` are labeled as nuclei.
        area_min (float): Minimum area for retaining nuclei after segmentation.
        area_max (float): Maximum area for retaining nuclei after segmentation.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map to foreground prior to watershedding. Default is 1.35.
        radius (float, optional): Radius of disk used in local mean thresholding to identify foreground. Default is 15.

    Returns:
        numpy.ndarray: Labeled segmentation mask of nuclei.
    """
    # Extract DAPI channel from the input data
    if isinstance(data, list):
        dapi = data[0]
    elif data.ndim == 3:
        dapi = data[0]
    else:
        dapi = data

    # Define keyword arguments for find_nuclei function
    kwargs = dict(
        threshold=lambda x: threshold,
        area_min=area_min,
        area_max=area_max,
        smooth=smooth,
        radius=radius,
    )

    # Suppress precision warning from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use find_nuclei function to segment nuclei from DAPI channel
        nuclei = find_nuclei(dapi, **kwargs)

    # Calculate the number of segmented nuclei (excluding background label)
    num_nuclei_segmented = len(np.unique(nuclei)) - 1
    print(f"Number of nuclei segmented: {num_nuclei_segmented}")

    # Convert nuclei array to uint16 dtype and return
    return nuclei.astype(np.uint16)


def segment_cells(data, nuclei, threshold, add_nuclei=True):
    """Segment cells from aligned data and match cell labels to nuclei labels.

    Note that labels can be skipped, for example if cells are touching the
    image boundary.

    Args:
        data : np.ndarray
            The aligned image data. Can have 2, 3, or 4 dimensions.
        nuclei : np.ndarray
            The segmented nuclei data.
        threshold : float
            The threshold value for cell segmentation.
        add_nuclei : bool, default True
            Whether to add the nuclei shape to the cell mask to help with mapping
            reads to cells at the edge of the field of view.

    Returns:
        np.ndarray
            The segmented cells, with labels matched to nuclei.
    """
    # Determine the mask based on the number of dimensions in data
    if data.ndim == 4:
        # If data has 4 dimensions: no DAPI, min over cycles, mean over channels
        mask = data[:, 1:].min(axis=0).mean(axis=0)
    elif data.ndim == 3:
        # If data has 3 dimensions: median over the remaining channels
        mask = np.median(data[1:], axis=0)
    elif data.ndim == 2:
        # If data has 2 dimensions: use the data directly as the mask
        mask = data
    else:
        # Raise an error if data has an unsupported number of dimensions
        raise ValueError("Data must have 2, 3, or 4 dimensions")

    # Apply the threshold to the mask to create a binary mask
    mask = mask > threshold

    # Add the nuclei to the mask if add_nuclei is True
    if add_nuclei:
        mask += nuclei.astype(bool)

    try:
        # Suppress skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Find cells in the mask
            cells = find_cells(nuclei, mask)
    except ValueError:
        # Handle the case where no cells are found
        print("segment_cells error -- no cells")
        cells = nuclei

    # Calculate the number of segmented cells (excluding background label)
    num_cells_segmented = len(np.unique(cells)) - 1
    print(f"Number of cells segmented: {num_cells_segmented}")

    # Return the segmented cells
    return cells


def find_cells(nuclei, mask, remove_boundary_cells=True):
    """Convert binary mask to cell labels, based on nuclei labels.

    Expands labeled nuclei to cells, constrained to where mask is >0.

    Args:
        nuclei (numpy.ndarray): Labeled segmentation mask of nuclei.
        mask (numpy.ndarray): Binary mask indicating valid regions for cell expansion.
        remove_boundary_cells (bool, optional): Whether to remove cells touching the boundary. Default is True.

    Returns:
        numpy.ndarray: Labeled segmentation mask of cells.
    """
    # Calculate distance transform of areas where nuclei are not present
    distance = ndi.distance_transform_cdt(nuclei == 0)

    # Use watershed segmentation to expand nuclei labels to cells within the mask
    cells = watershed(distance, nuclei, mask=mask)

    # Remove cells touching the boundary if specified
    if remove_boundary_cells:
        # Identify cells touching the boundary
        cut = np.concatenate([cells[0, :], cells[-1, :], cells[:, 0], cells[:, -1]])
        # Set labels of boundary-touching cells to 0
        cells.flat[np.in1d(cells, np.unique(cut))] = 0

    return cells.astype(np.uint16)


def find_nuclei(
    dapi,
    threshold,
    radius=15,
    area_min=50,
    area_max=500,
    score=lambda r: r.mean_intensity,
    smooth=1.35,
):
    """Segment nuclei from DAPI stain using various parameters and filters.

    Args:
        dapi (numpy.ndarray): Input DAPI image.
        threshold (float or callable): Threshold for mean intensity to segment nuclei.
                                     If callable, it should take an array of scores and return a threshold.
        radius (int, optional): Radius of disk used in local mean thresholding to identify foreground. Default is 15.
        area_min (int, optional): Minimum area for retaining nuclei after segmentation. Default is 50.
        area_max (int, optional): Maximum area for retaining nuclei after segmentation. Default is 500.
        score (function, optional): Function to calculate region score. Default is lambda r: r.mean_intensity.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map to foreground prior to watershedding. Default is 1.35.

    Returns:
        result (numpy.ndarray): Labeled segmentation mask of nuclei.
    """
    # Binarize DAPI image to identify foreground
    mask = binarize(dapi, radius, area_min)

    # Label connected components in the binary mask
    labeled = label(mask)

    # Filter labeled regions based on intensity score and threshold
    labeled = filter_by_region(labeled, score, threshold, intensity_image=dapi) > 0

    # Fill holes in the labeled mask
    filled = ndi.binary_fill_holes(labeled)

    # Label the differences between filled and original labeled regions
    difference = label(filled != labeled)

    # Identify regions with changes in area and update labeled mask
    change = filter_by_region(difference, lambda r: r.area < area_min, 0) > 0
    labeled[change] = filled[change]

    # Apply watershed algorithm to refine segmentation
    nuclei = apply_watershed(labeled, smooth=smooth)

    # Filter resulting nuclei by area range
    result = filter_by_region(nuclei, lambda r: area_min < r.area < area_max, threshold)

    return result


def filter_by_region(labeled, score, threshold, intensity_image=None, relabel=True):
    """Apply a filter to label image based on region properties.

    Args:
        labeled (numpy.ndarray): Labeled image.
        score (function): Function to calculate region score.
        threshold (float or callable): Threshold value to filter regions.
                                     If callable, it should take an array of scores and return a threshold.
        intensity_image (numpy.ndarray, optional): Intensity image for calculating scores. Default is None.
        relabel (bool, optional): Flag to relabel the regions sequentially. Default is True.

    Returns:
        labeled (numpy.ndarray): Filtered and relabeled image.
    """
    # Copy the labeled image to avoid modifying the original
    labeled = labeled.copy().astype(int)

    # Compute region properties
    regions = regionprops(labeled, intensity_image=intensity_image)

    # Calculate scores for each region
    scores = np.array([score(r) for r in regions])

    if all([s in (True, False) for s in scores]):
        # Identify regions to cut based on boolean scores
        cut = [r.label for r, s in zip(regions, scores) if not s]
    else:
        # Determine threshold value for scores
        if callable(threshold):
            t = threshold(scores)
        else:
            t = threshold
        cut = [r.label for r, s in zip(regions, scores) if s < t]

    # Remove identified regions from the labeled image
    labeled.flat[np.in1d(labeled.flat[:], cut)] = 0

    if relabel:
        # Relabel the regions sequentially
        labeled, _, _ = relabel_sequential(labeled)

    return labeled


def apply_watershed(img, smooth=4):
    """Apply the watershed algorithm to the given image to refine segmentation.

    Args:
        img (numpy.ndarray): Input binary image.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map. Default is 4.

    Returns:
        result (numpy.ndarray): Labeled image after watershed segmentation.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(img)

    if smooth > 0:
        # Apply Gaussian smoothing to the distance transform
        distance = gaussian(distance, sigma=smooth)

    # local_max = peak_local_max(
    #                 distance, indices=False, footprint=np.ones((3, 3)),
    #                 exclude_border=False)

    # Identify local maxima in the distance transform
    coordinates = peak_local_max(
        distance, min_distance=1, footprint=np.ones((3, 3)), exclude_border=False
    )

    # Create a boolean mask of local maxima
    local_max = np.zeros_like(distance, dtype=bool)
    if len(coordinates) > 0:
        local_max[tuple(coordinates.T)] = True

    # Label the local maxima
    markers = ndi.label(local_max)[0]

    # Apply watershed algorithm to the distance transform
    result = watershed(-distance, markers, mask=img)

    return result.astype(np.uint16)


def binarize(image, radius, min_size):
    """Apply local mean thresholding to binarize the image and remove small objects.

    Args:
        image (numpy.ndarray): Input image.
        radius (int): Radius of disk used in local mean thresholding.
        min_size (int): Minimum size of objects to retain.

    Returns:
        mask (numpy.ndarray): Binary mask of the image.
    """
    # Convert image to 8-bit unsigned integers
    dapi = img_as_ubyte(image)

    # Create a disk-shaped structuring element for filtering
    footprint = disk(radius)

    # Apply local mean filtering to the image
    mean_filtered = skimage.filters.rank.mean(dapi, footprint=footprint)

    # Create a binary mask by thresholding the image
    mask = dapi > mean_filtered

    # Remove small objects from the mask
    mask = remove_small_objects(mask, min_size=min_size)

    return mask
