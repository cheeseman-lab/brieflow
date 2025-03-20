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
from collections import defaultdict

from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    AutomaticMaskGenerator,
    get_predictor_and_decoder,
    mask_data_to_segmentation,
)
from skimage.measure import regionprops
from skimage.segmentation import clear_border


def segment_microsam(
    data,
    dapi_index,
    cyto_index,
    model_type="vit_b_lm",
    microsam_kwargs=dict(
        points_per_side=32,
        points_per_batch=16,
        stability_score_thresh=0.95,
        pred_iou_thresh=0.88,
    ),
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
    # Initialize microsam_kwargs if None
    if microsam_kwargs is None:
        microsam_kwargs = {}

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
                **microsam_kwargs,
            )
            counts.update(seg_counts)
        else:
            nuclei, cells = segment_microsam_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                gpu=gpu,
                **microsam_kwargs,
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
        nuclei = segment_microsam_nuclei(dapi, model_type=model_type, **microsam_kwargs)

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
    **kwargs,
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
        **kwargs: Additional keyword arguments for MicroSAM segmentation

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
    dapi, model_type="vit_b_lm", remove_edges=True, gpu=False, **kwargs
):
    """Segment nuclei using the MicroSAM algorithm with InstanceSegmentationWithDecoder.

    Args:
        dapi (numpy.ndarray): DAPI channel image
        model_type (str, optional): MicroSAM model type to use
        remove_edges (bool, optional): Whether to remove nuclei touching the image edges
        gpu (bool, optional): Whether to use GPU for segmentation
        **kwargs: Additional keyword arguments for MicroSAM segmentation

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
        regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True
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
