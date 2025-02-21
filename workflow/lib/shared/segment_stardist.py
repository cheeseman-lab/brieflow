"""StarDist-based Image Segmentation

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
from collections import defaultdict
from typing import Tuple, Dict, Optional, Union

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops
from skimage.segmentation import clear_border


def segment_stardist(
    data: np.ndarray,
    dapi_index: int,
    cyto_index: int,
    model_type: str = "2D_versatile_fluo",
    stardist_kwargs: dict = dict(
        prob_thresh=0.5,
        nms_thresh=0.4,
    ),
    cells: bool = True,
    reconcile: str = "consensus",
    # logscale: bool = True,
    return_counts: bool = False,
    gpu: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, pd.DataFrame], 
           Tuple[np.ndarray, np.ndarray], 
           np.ndarray, 
           Tuple[np.ndarray, pd.DataFrame]]:
    """Segment cells using StarDist algorithm.
    
    Args:
        data: Multichannel image data
        dapi_index: Index of DAPI channel
        cyto_index: Index of cytoplasmic channel
        nuclei_diameter: Unused but kept for interface consistency
        cell_diameter: Unused but kept for interface consistency
        model_type: StarDist model type to use
        stardist_kwargs: Additional keyword arguments for StarDist segmentation
        cells: Whether to segment both nuclei and cells or just nuclei
        reconcile: Method for reconciling nuclei and cells
        logscale: Whether to apply logarithmic transformation to image data
        return_counts: Whether to return counts of nuclei and cells
        gpu: Whether to use GPU for segmentation
    
    Returns:
        Segmentation masks with optional counts
    """
    # Prepare channels for StarDist
    dapi = prepare_channel(data[dapi_index])
    cyto = prepare_channel(data[cyto_index])
    counts = {}

    # Perform cell segmentation
    if cells:
        if return_counts:
            nuclei, cells, seg_counts = segment_stardist_multichannel(
                dapi, 
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
                **stardist_kwargs
            )
            counts.update(seg_counts)
        else:
            nuclei, cells = segment_stardist_multichannel(
                dapi, 
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                gpu=gpu,
                **stardist_kwargs
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
            dapi, 
            model_type=model_type,
            gpu=gpu,
            **stardist_kwargs
        )
        
        counts["final_nuclei"] = len(np.unique(nuclei)) - 1
        print(f"Number of nuclei segmented: {counts['final_nuclei']}")
        
        counts_df = pd.DataFrame([counts])
        
        if return_counts:
            return nuclei, counts_df
        else:
            return nuclei


def prepare_channel(
    data: np.ndarray,
) -> np.ndarray:
    """Prepare channel data for segmentation using StarDist's normalization.
    
    Args:
        data: Input channel data
        
    Returns:
        Processed channel data
    """
    # Use StarDist's recommended normalization
    return normalize(data, 1, 99.8, axis=None)



def segment_stardist_multichannel(
    dapi: np.ndarray,
    cyto: np.ndarray,
    model_type: str = "2D_versatile_fluo",
    reconcile: str = "consensus",
    remove_edges: bool = True,
    return_counts: bool = False,
    gpu: bool = False,
    **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict]]:
    """Segment nuclei and cells using the StarDist algorithm.
    
    Args:
        dapi: DAPI channel image
        cyto: Cytoplasmic channel image
        model_type: StarDist model type to use
        reconcile: Method for reconciling nuclei and cells
        remove_edges: Whether to remove nuclei and cells touching image edges
        return_counts: Whether to return counts of nuclei and cells
        gpu: Whether to use GPU for segmentation
        **kwargs: Additional keyword arguments for StarDist segmentation
    
    Returns:
        Segmentation masks with optional counts
    """
    counts = {}
    
    # Initialize StarDist models for nuclei and cytoplasmic segmentation
    model_nuclei = StarDist2D.from_pretrained(model_type)
    model_cells = StarDist2D.from_pretrained(model_type)
    
    if gpu:
        model_nuclei.config.use_gpu = True
        model_cells.config.use_gpu = True
    
    # Segment nuclei and cells using StarDist
    print("Performing StarDist nuclear segmentation...", file=sys.stderr)
    nuclei, _ = model_nuclei.predict_instances(dapi, **kwargs)
    
    print("Performing StarDist cell segmentation...", file=sys.stderr)
    cells, _ = model_cells.predict_instances(cyto, **kwargs)
    
    counts["initial_nuclei"] = len(np.unique(nuclei)) - 1
    counts["initial_cells"] = len(np.unique(cells)) - 1
    
    print(
        f'found {counts["initial_nuclei"]} nuclei before removing edges',
        file=sys.stderr,
    )
    print(
        f'found {counts["initial_cells"]} cells before removing edges', 
        file=sys.stderr
    )
    
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
    
    if reconcile:
        print(f"reconciling masks with method how={reconcile}")
        nuclei, cells = reconcile_nuclei_cells(nuclei, cells, how=reconcile)
    
    counts["final_cells"] = len(np.unique(cells)) - 1
    
    print(
        f'found {counts["final_cells"]} nuclei/cells after reconciling', 
        file=sys.stderr
    )
    
    if return_counts:
        return nuclei, cells, counts
    else:
        return nuclei, cells


def segment_stardist_nuclei(
    dapi: np.ndarray,
    model_type: str = "2D_versatile_fluo",
    remove_edges: bool = True,
    gpu: bool = False,
    **kwargs
) -> np.ndarray:
    """Segment nuclei using the StarDist algorithm.
    
    Args:
        dapi: DAPI channel image
        model_type: StarDist model type to use
        remove_edges: Whether to remove nuclei touching the image edges
        gpu: Whether to use GPU for segmentation
        **kwargs: Additional keyword arguments for StarDist segmentation
    
    Returns:
        Labeled segmentation mask of nuclei
    """
    # Initialize StarDist model
    model = StarDist2D.from_pretrained(model_type)
    if gpu:
        model.config.use_gpu = True
    
    # Segment nuclei
    print("Performing StarDist segmentation...", file=sys.stderr)
    nuclei, _ = model.predict_instances(dapi, **kwargs)
    
    print(
        f"found {len(np.unique(nuclei))} nuclei before removing edges", 
        file=sys.stderr
    )
    
    if remove_edges:
        print("removing edges")
        nuclei = clear_border(nuclei)
    
    print(f"found {len(np.unique(nuclei))} final nuclei", file=sys.stderr)
    
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

    # Always get the multiple nuclei mapping for analysis
    cell_map_multiple = get_unique_label_map(
        regionprops(cells, intensity_image=nuclei_eroded), 
        keep_multiple=True
    )
    
    # Count cells with multiple nuclei
    nuclei_per_cell = defaultdict(int)
    for cell_label, nuclei_labels in cell_map_multiple.items():
        nuclei_per_cell[len(nuclei_labels)] += 1
    
    # Print statistics
    print("\nNuclei per cell statistics:")
    print("--------------------------")
    for num_nuclei, count in sorted(nuclei_per_cell.items()):
        print(f"Cells with {num_nuclei} nuclei: {count}")
    print("--------------------------\n")

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
