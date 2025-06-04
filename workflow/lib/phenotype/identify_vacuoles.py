"""Segment vacuoles using skimage thresholding and visualize vacuoles using microfilm.

This module provides functions for segmenting and visualizing vacuoles in microscopy images.
It includes functions for:

1. Vacuole Segmentation: Segmenting vacuoles within cells based on thresholding.
2. Nuclei Detection: Identifying nuclei/intensity peaks within vacuoles.
3. Cell-Vacuole Association: Mapping vacuoles to their containing cells.
4. Cytoplasm Adjustment: Updating cytoplasm masks by removing vacuole regions.
5. Visualization: Creating enhanced visualizations of cells, vacuoles, and detected nuclei.

"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation, feature, util, exposure
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from microfilm.microplot import Microimage

from lib.shared.configuration_utils import create_micropanel


import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, feature, measure, segmentation, exposure


def segment_vacuoles(
    image,
    vacuole_channel_index,
    nuclei_channel_index=None,
    cell_masks=None,
    cytoplasm_masks=None,
    min_size=20,
    max_size=5000,
    threshold_smoothing_scale=1.3488,
    min_distance_between_maxima=20,
    max_objects_per_cell=120,
    overlap_threshold=0.1,
    nuclei_min_distance=5,
    nuclei_centroids=None,
):
    """Segment vacuoles within cells using thresholding and declumping.

    Args:
        image (numpy.ndarray): Multichannel image data with shape [channels, height, width].
        vacuole_channel_index (int): Index of the channel used for vacuole detection.
        nuclei_channel_index (int, optional): Index of the channel for detecting nuclei/peaks within vacuoles.
            If None, the same channel as vacuole_channel_index will be used.
        cell_masks (numpy.ndarray): Cell segmentation masks with unique integers for each cell.
        cytoplasm_masks (numpy.ndarray, optional): Cytoplasm segmentation masks with unique integers.
            If provided, vacuole regions will be removed from cytoplasm masks.
        min_size (int, optional): Minimum size (in pixels) for a vacuole to be considered valid. Default is 20.
        max_size (int, optional): Maximum size (in pixels) for a vacuole to be considered valid. Default is 5000.
        threshold_smoothing_scale (float, optional): Sigma for Gaussian smoothing before thresholding. Default is 1.3488.
        min_distance_between_maxima (int, optional): Minimum distance between local maxima for declumping vacuoles. Default is 20.
        max_objects_per_cell (int, optional): Maximum number of vacuoles allowed per cell. Default is 120.
        overlap_threshold (float, optional): Minimum overlap ratio required to associate a vacuole with a cell. Default is 0.1.
        nuclei_min_distance (int, optional): Minimum distance between peaks when identifying nuclei within vacuoles. Default is 5.
        nuclei_centroids (dict or pandas.DataFrame, optional): Dictionary or DataFrame containing nuclei centroids
            with keys/columns 'i', 'j' for y and x coordinates. Used to calculate distance from vacuoles to nuclei.

    Returns:
        tuple: A tuple containing:
            - vacuole_masks (numpy.ndarray): Labeled mask of vacuoles with their original unique IDs.
            - cell_vacuole_table (dict): Dictionary with DataFrames containing cell-vacuole associations and measurements.
            - (optional) updated_cytoplasm_masks (numpy.ndarray): Updated cytoplasm masks with vacuole regions removed.
              Only returned if cytoplasm_masks is provided.
    """
    # Extract the vacuole channel
    vacuole_img = image[vacuole_channel_index]

    # Apply log transform and smoothing
    vacuole_log = exposure.adjust_log(vacuole_img)
    vacuole_smooth = filters.gaussian(vacuole_log, sigma=threshold_smoothing_scale)

    # Apply Otsu thresholding
    thresh = filters.threshold_otsu(vacuole_smooth)
    binary_mask = vacuole_smooth > thresh
    filled_mask = ndimage.binary_fill_holes(binary_mask)

    # Early exit if no objects found
    if not np.any(filled_mask):
        print("No objects detected after thresholding")
        return _create_empty_results(cell_masks, cytoplasm_masks)

    # Declumping with watershed
    distance = ndimage.distance_transform_edt(filled_mask)
    local_max = feature.peak_local_max(
        distance, min_distance=min_distance_between_maxima, labels=filled_mask
    )

    # Create markers and apply watershed
    markers = np.zeros_like(filled_mask, dtype=int)
    if len(local_max) > 0:
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        declumped = segmentation.watershed(-distance, markers, mask=filled_mask)
    else:
        declumped, _ = ndimage.label(filled_mask)

    # Fill holes after declumping - OPTIMIZED: vectorized approach
    unique_labels = np.unique(declumped[declumped > 0])
    for label in unique_labels:
        mask = declumped == label
        filled = ndimage.binary_fill_holes(mask)
        declumped[filled] = label

    # OPTIMIZED: Vectorized size filtering
    regions = measure.regionprops(declumped)
    valid_labels = [region.label for region in regions if min_size <= region.area <= max_size]
    
    if not valid_labels:
        print("No valid vacuoles found after size filtering")
        return _create_empty_results(cell_masks, cytoplasm_masks)
    
    # Create valid vacuoles mask efficiently
    valid_vacuoles = np.isin(declumped, valid_labels) * declumped
    labeled_vacuoles, num_vacuoles = ndimage.label(valid_vacuoles > 0)

    # Get cell IDs once
    cell_ids = np.unique(cell_masks[cell_masks > 0])

    # Determine nuclei channel
    nuclei_channel_index = nuclei_channel_index or vacuole_channel_index
    nuclei_img = image[nuclei_channel_index]

    # Prepare nuclei centroids - OPTIMIZED: do this once
    nuclei_centroids_dict = None
    if nuclei_centroids is not None:
        if isinstance(nuclei_centroids, pd.DataFrame):
            nuclei_centroids_dict = {}
            for idx, row in nuclei_centroids.iterrows():
                nuc_id = row.get("nuclei_id", idx)
                nuclei_centroids_dict[nuc_id] = (row["i"], row["j"])
        else:
            nuclei_centroids_dict = nuclei_centroids

    # OPTIMIZED: Pre-compute region properties for all vacuoles
    vacuole_regions = {region.label: region for region in measure.regionprops(labeled_vacuoles)}

    # Initialize tracking variables
    vacuole_cell_mapping = []
    vacuoles_per_cell = {cell_id: 0 for cell_id in cell_ids}
    nuclei_per_vacuole = {}

    # Process each vacuole
    for vacuole_id in range(1, num_vacuoles + 1):
        if vacuole_id not in vacuole_regions:
            continue
            
        region = vacuole_regions[vacuole_id]
        vacuole_mask = labeled_vacuoles == vacuole_id
        vacuole_area = region.area
        vacuole_centroid = region.centroid

        # Find nuclei peaks within this vacuole
        peaks = feature.peak_local_max(
            nuclei_img,
            min_distance=nuclei_min_distance,
            labels=vacuole_mask,
            exclude_border=False,
        )

        nuclei_per_vacuole[vacuole_id] = {
            "count": len(peaks),
            "peak_coordinates": peaks.tolist() if len(peaks) > 0 else [],
        }

        # Calculate distance to nearest nucleus centroid
        min_distance_to_nucleus = None
        nearest_nucleus_id = None

        if nuclei_centroids_dict is not None:
            min_dist = np.inf
            for nuc_id, nuc_centroid in nuclei_centroids_dict.items():
                dist = np.sqrt(
                    (vacuole_centroid[0] - nuc_centroid[0]) ** 2
                    + (vacuole_centroid[1] - nuc_centroid[1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_nucleus_id = nuc_id
            
            min_distance_to_nucleus = min_dist if min_dist != np.inf else None

        # OPTIMIZED: Find best overlapping cell more efficiently
        best_cell_id = None
        best_overlap = 0

        for cell_id in cell_ids:
            if vacuoles_per_cell[cell_id] >= max_objects_per_cell:
                continue

            # Calculate overlap efficiently
            cell_mask = cell_masks == cell_id
            overlap = np.sum(vacuole_mask & cell_mask)
            
            if overlap > 0:
                overlap_ratio = overlap / vacuole_area
                if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cell_id = cell_id

        # Add successful associations
        if best_cell_id is not None:
            vacuole_cell_mapping.append({
                "vacuole_id": vacuole_id,
                "cell_id": best_cell_id,
                "vacuole_area": vacuole_area,
                "overlap_ratio": best_overlap,
                "nuclei_count": nuclei_per_vacuole[vacuole_id]["count"],
                "peak_coordinates": nuclei_per_vacuole[vacuole_id]["peak_coordinates"],
                "distance_to_nucleus": min_distance_to_nucleus,
                "nearest_nucleus_id": nearest_nucleus_id,
            })
            vacuoles_per_cell[best_cell_id] += 1

    # Create vacuole-cell mapping DataFrame
    vacuole_cell_df = pd.DataFrame(vacuole_cell_mapping)

    # OPTIMIZED: Create cell summary more efficiently
    if vacuole_cell_mapping:
        # Group by cell_id once for efficiency
        grouped = vacuole_cell_df.groupby('cell_id') if not vacuole_cell_df.empty else None
        
        cell_summary = []
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            
            if grouped is not None and cell_id in grouped.groups:
                cell_vacuoles = grouped.get_group(cell_id)
                total_vacuole_area = cell_vacuoles["vacuole_area"].sum()
                total_nuclei = cell_vacuoles["nuclei_count"].sum()
                multinucleated_count = len(cell_vacuoles[cell_vacuoles["nuclei_count"] > 1])
                
                # Calculate mean distance to nucleus
                valid_distances = cell_vacuoles["distance_to_nucleus"].dropna()
                mean_distance = valid_distances.mean() if len(valid_distances) > 0 else None
                
                cell_summary.append({
                    "cell_id": cell_id,
                    "has_vacuole": True,
                    "num_vacuoles": len(cell_vacuoles),
                    "vacuole_ids": list(cell_vacuoles["vacuole_id"]),
                    "cell_area": cell_area,
                    "total_vacuole_area": total_vacuole_area,
                    "vacuole_area_ratio": total_vacuole_area / cell_area if cell_area > 0 else 0,
                    "total_nuclei_in_vacuoles": total_nuclei,
                    "multinucleated_vacuole_count": multinucleated_count,
                    "mean_distance_to_nucleus": mean_distance,
                })
            else:
                cell_summary.append({
                    "cell_id": cell_id,
                    "has_vacuole": False,
                    "num_vacuoles": 0,
                    "vacuole_ids": [],
                    "cell_area": cell_area,
                    "total_vacuole_area": 0,
                    "vacuole_area_ratio": 0,
                    "total_nuclei_in_vacuoles": 0,
                    "multinucleated_vacuole_count": 0,
                    "mean_distance_to_nucleus": None,
                })
    else:
        # Handle case with no vacuoles
        cell_summary = []
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            cell_summary.append({
                "cell_id": cell_id,
                "has_vacuole": False,
                "num_vacuoles": 0,
                "vacuole_ids": [],
                "cell_area": cell_area,
                "total_vacuole_area": 0,
                "vacuole_area_ratio": 0,
                "total_nuclei_in_vacuoles": 0,
                "multinucleated_vacuole_count": 0,
                "mean_distance_to_nucleus": None,
            })

    # Create final results
    cell_summary_df = pd.DataFrame(cell_summary)
    cell_vacuole_table = {
        "cell_summary": cell_summary_df,
        "vacuole_cell_mapping": vacuole_cell_df,
    }

    # Create associated vacuole masks
    associated_vacuoles = np.zeros_like(labeled_vacuoles)
    for mapping in vacuole_cell_mapping:
        vacuole_id = mapping["vacuole_id"]
        vacuole_mask = labeled_vacuoles == vacuole_id
        associated_vacuoles[vacuole_mask] = vacuole_id

    # Print statistics
    total_kept = len(vacuole_cell_mapping)
    print(f"Kept {total_kept} out of {num_vacuoles} detected vacuoles "
          f"({total_kept / num_vacuoles * 100:.1f}%)")
    print(f"Discarded {num_vacuoles - total_kept} vacuoles that didn't sufficiently overlap with cells")

    # Process cytoplasm masks if provided
    updated_cytoplasm_masks = None
    if cytoplasm_masks is not None:
        updated_cytoplasm_masks = cytoplasm_masks.copy()
        
        for mapping in vacuole_cell_mapping:
            vacuole_id = mapping["vacuole_id"]
            cell_id = mapping["cell_id"]
            
            vacuole_mask = associated_vacuoles == vacuole_id
            cytoplasm_mask = updated_cytoplasm_masks == cell_id
            updated_cytoplasm_masks[cytoplasm_mask & vacuole_mask] = 0

        print(f"Updated cytoplasm masks by removing {len(vacuole_cell_mapping)} vacuole regions")

    # Return results
    if updated_cytoplasm_masks is not None:
        return associated_vacuoles, cell_vacuole_table, updated_cytoplasm_masks
    else:
        return associated_vacuoles, cell_vacuole_table


def _create_empty_results(cell_masks, cytoplasm_masks):
    """Helper function to create empty results when no vacuoles are found."""
    cell_ids = np.unique(cell_masks[cell_masks > 0])
    empty_vacuole_masks = np.zeros_like(cell_masks)
    
    cell_summary = []
    for cell_id in cell_ids:
        cell_area = np.sum(cell_masks == cell_id)
        cell_summary.append({
            "cell_id": cell_id,
            "has_vacuole": False,
            "num_vacuoles": 0,
            "vacuole_ids": [],
            "cell_area": cell_area,
            "total_vacuole_area": 0,
            "vacuole_area_ratio": 0,
            "total_nuclei_in_vacuoles": 0,
            "multinucleated_vacuole_count": 0,
            "mean_distance_to_nucleus": None,
        })
    
    cell_vacuole_table = {
        "cell_summary": pd.DataFrame(cell_summary),
        "vacuole_cell_mapping": pd.DataFrame()
    }
    
    if cytoplasm_masks is not None:
        return empty_vacuole_masks, cell_vacuole_table, cytoplasm_masks
    else:
        return empty_vacuole_masks, cell_vacuole_table

def create_vacuole_boundary_visualization(
    image,
    vacuole_channel_index,
    cell_masks,
    vacuole_masks,
    vacuole_cell_mapping=None,
    channel_names=None,
    channel_cmaps=None,
    show_nuclei_peaks=True,
):
    """Create enhanced visualization showing cells, vacuoles, and detected nuclei.

    Args:
        image (numpy.ndarray): Multichannel image data with shape [channels, height, width].
        vacuole_channel_index (int): Index of the channel used for vacuole detection.
        cell_masks (numpy.ndarray): Cell segmentation masks with unique integers for each cell.
        vacuole_masks (numpy.ndarray): Vacuole segmentation masks with original vacuole IDs.
        vacuole_cell_mapping (pandas.DataFrame, optional): DataFrame containing mapping between vacuoles and cells,
            including peak coordinates.
        channel_names (list of str, optional): Names for each channel in the image.
        channel_cmaps (list of str, optional): Color maps for each channel in the image.
        show_nuclei_peaks (bool, optional): Whether to show detected nuclei/peaks on the visualization. Default is True.

    Returns:
        matplotlib.figure.Figure: The created micropanel figure showing the cell boundaries (green),
            vacuole boundaries (magenta), and nuclei/peak markers (yellow) overlaid on the image.
    """
    if channel_names is None or len(channel_names) <= vacuole_channel_index:
        channel_name = f"Channel {vacuole_channel_index}"
    else:
        channel_name = channel_names[vacuole_channel_index]

    # Get vacuole channel
    vacuole_img = image[vacuole_channel_index].copy()

    # Create a copy of the original image for the merged view with boundaries
    merged_img = image.copy()

    # Function to add boundaries and peaks to an image
    def add_boundaries_and_peaks(base_image, base_is_multichannel=True):
        # Determine the shape based on whether base_image is multichannel or single channel
        if base_is_multichannel:
            # For multichannel image, keep as is
            enhanced_img = base_image.copy()
            height, width = base_image.shape[1], base_image.shape[2]
            num_channels = base_image.shape[0]
        else:
            # For single channel image, expand to 3 channels
            height, width = base_image.shape
            num_channels = 3
            # Create 3-channel image with the base image in all channels
            enhanced_img = np.zeros((num_channels, height, width), dtype=np.float32)
            base_norm = base_image / (base_image.max() if base_image.max() > 0 else 1.0)
            for c in range(num_channels):
                enhanced_img[c] = base_norm

        # Add cell boundaries (green)
        if base_is_multichannel:
            # For multichannel image, we need to create a temporary RGB image
            # to use mark_boundaries, then extract the green channel
            temp_img = np.zeros((height, width, 3), dtype=np.float32)
            for c in range(min(3, num_channels)):
                temp_img[:, :, c] = enhanced_img[c] / (
                    enhanced_img[c].max() if enhanced_img[c].max() > 0 else 1.0
                )

            cell_boundary_img = mark_boundaries(
                temp_img,
                cell_masks,
                color=(0, 1, 0),  # Green for cells
                mode="thick",
            )

            # Update the green channel with cell boundaries - make them more prominent
            cell_boundary_intensity = (
                1.2 * enhanced_img[1].max()
            )  # Increase intensity by 20%
            enhanced_img[1] = np.maximum(
                enhanced_img[1], cell_boundary_img[:, :, 1] * cell_boundary_intensity
            )
            # Cap values at 1.0 if normalized
            if enhanced_img.dtype == np.float32 or enhanced_img.dtype == np.float64:
                enhanced_img[1] = np.minimum(
                    enhanced_img[1],
                    1.0 if enhanced_img[1].max() <= 1.0 else enhanced_img[1].max(),
                )
        else:
            # For single channel image, directly add boundaries to green channel
            cell_boundary = mark_boundaries(
                base_image,
                cell_masks,
                color=(0, 1, 0),  # Green for cells
                mode="thick",
            )
            enhanced_img[1] = np.maximum(enhanced_img[1], cell_boundary[:, :, 1])

        # Add vacuole boundaries (magenta: red + blue)
        if base_is_multichannel:
            # For multichannel image, create temporary RGB again
            vacuole_boundary_img = mark_boundaries(
                temp_img,
                vacuole_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for vacuoles
                mode="thick",
            )

            # Update red and blue channels with vacuole boundaries
            enhanced_img[0] = np.maximum(
                enhanced_img[0], vacuole_boundary_img[:, :, 0] * enhanced_img[0].max()
            )
            if num_channels > 2:  # Make sure we have a blue channel
                enhanced_img[2] = np.maximum(
                    enhanced_img[2],
                    vacuole_boundary_img[:, :, 2] * enhanced_img[2].max(),
                )
        else:
            # For single channel, add boundaries to red and blue channels
            vacuole_boundary = mark_boundaries(
                base_image,
                vacuole_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for vacuoles
                mode="thick",
            )
            enhanced_img[0] = np.maximum(enhanced_img[0], vacuole_boundary[:, :, 0])
            enhanced_img[2] = np.maximum(enhanced_img[2], vacuole_boundary[:, :, 2])

        # Add nuclei/peak markers if requested and available
        if (
            show_nuclei_peaks
            and vacuole_cell_mapping is not None
            and "peak_coordinates" in vacuole_cell_mapping.columns
        ):
            for _, row in vacuole_cell_mapping.iterrows():
                if "peak_coordinates" in row and row["peak_coordinates"]:
                    for coord in row["peak_coordinates"]:
                        # Make sure coordinates are valid
                        if len(coord) >= 2:
                            y, x = coord[0], coord[1]
                            # Ensure coordinates are within image bounds
                            if 0 <= y < height and 0 <= x < width:
                                # Draw a 3x3 yellow dot
                                for dy in range(-1, 2):
                                    for dx in range(-1, 2):
                                        ny, nx = y + dy, x + dx
                                        if 0 <= ny < height and 0 <= nx < width:
                                            # Yellow = Red + Green
                                            max_val = (
                                                enhanced_img[:, ny, nx].max()
                                                if base_is_multichannel
                                                else 1.0
                                            )
                                            enhanced_img[0, ny, nx] = max_val  # Red
                                            enhanced_img[1, ny, nx] = max_val  # Green
                                            if (
                                                num_channels > 2
                                            ):  # Make sure we have a blue channel
                                                enhanced_img[2, ny, nx] = 0.0  # No blue

        return enhanced_img

    # Create merged microimage with boundaries and peaks
    merged_with_boundaries = add_boundaries_and_peaks(merged_img)
    merged_microimage = Microimage(
        merged_with_boundaries, channel_names="Merged", cmaps=channel_cmaps
    )

    # Create vacuole channel microimage with boundaries and peaks
    # Convert single channel to 3D for processing
    vacuole_3d = add_boundaries_and_peaks(vacuole_img, base_is_multichannel=False)
    boundaries_microimage = Microimage(
        vacuole_3d,
        channel_names=f"{channel_name}",
        cmaps=["pure_red", "pure_green", "pure_blue"],
    )

    # Create the micropanel
    microimages = [merged_microimage, boundaries_microimage]
    panel = create_micropanel(microimages, add_channel_label=True)

    return panel
