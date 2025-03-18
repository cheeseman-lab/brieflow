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
    nuclei_centroids=None,  # Parameter for nuclei centroids from phenotype_minimal (i,j coordinates)
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
    vacuole_img = image[vacuole_channel_index].copy()
    
    # Apply log transform as specified
    vacuole_log = exposure.adjust_log(vacuole_img)
    
    # Apply Gaussian smoothing with specified sigma
    vacuole_smooth = filters.gaussian(vacuole_log, sigma=threshold_smoothing_scale)
    
    # Apply Otsu thresholding with two classes
    thresh = filters.threshold_otsu(vacuole_smooth)
    binary_mask = vacuole_smooth > thresh
    
    # Fill holes after thresholding
    filled_mask = ndimage.binary_fill_holes(binary_mask)
    
    # Declumping based on shape
    # 1. Distance transform
    distance = ndimage.distance_transform_edt(filled_mask)
    
    # 2. Find local maxima with minimum distance constraint
    local_max = feature.peak_local_max(
        distance, 
        min_distance=min_distance_between_maxima,
        labels=filled_mask
    )
    
    # 3. Create markers for watershed
    markers = np.zeros_like(filled_mask, dtype=int)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    
    # 4. Apply watershed to declump
    declumped = segmentation.watershed(-distance, markers, mask=filled_mask)
    
    # Fill holes after declumping
    for label in range(1, declumped.max() + 1):
        mask = declumped == label
        filled = ndimage.binary_fill_holes(mask)
        declumped[filled] = label
    
    # Measure region properties
    regions = measure.regionprops(declumped)
    
    # Filter by size
    valid_vacuoles = np.zeros_like(declumped)
    for region in regions:
        if min_size <= region.area <= max_size:
            valid_vacuoles[declumped == region.label] = region.label
            
    # Re-label the valid vacuoles
    if np.any(valid_vacuoles):
        labeled_vacuoles, num_vacuoles = ndimage.label(valid_vacuoles > 0)
    else:
        labeled_vacuoles = np.zeros_like(valid_vacuoles)
        num_vacuoles = 0
    
    # Get cell IDs
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids > 0]  # Remove background (0)
    
    # Initialize vacuole masks (will keep original vacuole IDs)
    vacuole_masks = labeled_vacuoles.copy()
    
    # List to store vacuole-cell mapping data
    vacuole_cell_mapping = []
    
    # Keep track of vacuoles per cell
    vacuoles_per_cell = {cell_id: 0 for cell_id in cell_ids}
    
    # Identify intensity peaks within vacuoles (similar to IdentifyPrimaryObjects)
    nuclei_per_vacuole = {}
    
    # Determine which channel to use for nuclei detection
    if nuclei_channel_index is None:
        nuclei_channel_index = vacuole_channel_index
    
    # Get the nuclei channel image
    nuclei_img = image[nuclei_channel_index].copy()
    
    # Prepare nuclei centroids if provided
    if nuclei_centroids is not None:
        # Convert to dictionary format if it's a DataFrame
        if isinstance(nuclei_centroids, pd.DataFrame):
            # Assuming the DataFrame has a column for nuclei IDs (adjust if needed)
            nuc_centroid_dict = {}
            for idx, row in nuclei_centroids.iterrows():
                # Assuming 'i' is y-coordinate and 'j' is x-coordinate
                # Use index as ID if no explicit ID column exists
                nuc_id = idx
                if 'nuclei_id' in row:
                    nuc_id = row['nuclei_id']
                nuc_centroid_dict[nuc_id] = (row['i'], row['j'])
            nuclei_centroids = nuc_centroid_dict
    
    # For each vacuole, identify intensity peaks
    for vacuole_id in range(1, num_vacuoles + 1):
        vacuole_mask = labeled_vacuoles == vacuole_id
        
        if np.sum(vacuole_mask) == 0:
            # Skip empty vacuoles
            nuclei_per_vacuole[vacuole_id] = {
                'count': 0,
                'peak_coordinates': []
            }
            continue
        
        # Mask the nuclei image with this vacuole
        masked_nuclei_img = np.zeros_like(nuclei_img)
        masked_nuclei_img[vacuole_mask] = nuclei_img[vacuole_mask]
        
        # Find local maxima within this vacuole
        # Use minimum_distance to control sensitivity
        peaks = feature.peak_local_max(
            masked_nuclei_img,
            min_distance=nuclei_min_distance,
            labels=vacuole_mask,
            exclude_border=False
        )
        
        # Store the count and coordinates of peaks for this vacuole
        nuclei_per_vacuole[vacuole_id] = {
            'count': len(peaks),
            'peak_coordinates': peaks.tolist() if len(peaks) > 0 else []
        }
    
    # Associate vacuoles with cells
    # Create a mask to track which vacuoles are assigned to cells
    associated_vacuoles = np.zeros_like(labeled_vacuoles)
    
    for vacuole_id in range(1, num_vacuoles + 1):
        vacuole_mask = labeled_vacuoles == vacuole_id
        vacuole_area = np.sum(vacuole_mask)
        
        # Calculate vacuole centroid
        vacuole_props = measure.regionprops(vacuole_mask.astype(int))
        if len(vacuole_props) > 0:
            vacuole_centroid = vacuole_props[0].centroid
        else:
            # If regionprops fails, calculate centroid manually
            y_indices, x_indices = np.where(vacuole_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                vacuole_centroid = (np.mean(y_indices), np.mean(x_indices))
            else:
                vacuole_centroid = None
        
        # Calculate distance to nearest nucleus centroid if nuclei_centroids provided
        min_distance_to_nucleus = np.inf
        nearest_nucleus_id = None
        
        if nuclei_centroids is not None and vacuole_centroid is not None:
            for nuc_id, nuc_centroid in nuclei_centroids.items():
                # Calculate Euclidean distance
                dist = np.sqrt((vacuole_centroid[0] - nuc_centroid[0])**2 + 
                               (vacuole_centroid[1] - nuc_centroid[1])**2)
                
                if dist < min_distance_to_nucleus:
                    min_distance_to_nucleus = dist
                    nearest_nucleus_id = nuc_id
        
        # If no nucleus was found, set to None
        if min_distance_to_nucleus == np.inf:
            min_distance_to_nucleus = None
        
        # Find overlapping cells
        overlapping_cells = {}
        for cell_id in cell_ids:
            cell_mask = cell_masks == cell_id
            
            # Skip cells that already have too many vacuoles
            if vacuoles_per_cell[cell_id] >= max_objects_per_cell:
                continue
                
            overlap = np.sum(vacuole_mask & cell_mask)
            if overlap > 0:
                overlap_ratio = overlap / vacuole_area
                overlapping_cells[cell_id] = overlap_ratio
        
        # Assign vacuole to the cell with maximum overlap if it meets the threshold
        if overlapping_cells:
            max_cell_id = max(overlapping_cells, key=overlapping_cells.get)
            max_overlap = overlapping_cells[max_cell_id]
            
            if max_overlap >= overlap_threshold:
                # Add this mapping to our list
                vacuole_cell_mapping.append({
                    "vacuole_id": vacuole_id,
                    "cell_id": max_cell_id,
                    "vacuole_area": vacuole_area,
                    "overlap_ratio": max_overlap,
                    "nuclei_count": nuclei_per_vacuole.get(vacuole_id, {}).get('count', 0),
                    "peak_coordinates": nuclei_per_vacuole.get(vacuole_id, {}).get('peak_coordinates', []),
                    "distance_to_nucleus": min_distance_to_nucleus,  # Added distance to nucleus
                    "nearest_nucleus_id": nearest_nucleus_id  # Added nearest nucleus ID
                })
                
                # Mark this vacuole as associated with a cell
                associated_vacuoles[vacuole_mask] = vacuole_id
                
                # Increment the vacuole count for this cell
                vacuoles_per_cell[max_cell_id] += 1
    
    # Create the vacuole-cell mapping DataFrame
    vacuole_cell_df = pd.DataFrame(vacuole_cell_mapping)
    
    # Create a cell summary DataFrame
    if vacuole_cell_mapping and 'cell_id' in vacuole_cell_df.columns:
        cell_summary = []
        for cell_id in cell_ids:
            # Get all vacuoles for this cell
            cell_vacuoles = vacuole_cell_df[vacuole_cell_df["cell_id"] == cell_id]
            
            # Calculate cell area
            cell_mask = cell_masks == cell_id
            cell_area = np.sum(cell_mask)
            
            # Get total vacuole area for this cell
            total_vacuole_area = cell_vacuoles["vacuole_area"].sum() if not cell_vacuoles.empty else 0
            
            # Calculate mean distance to nucleus for this cell's vacuoles
            mean_distance_to_nucleus = None
            if not cell_vacuoles.empty and 'distance_to_nucleus' in cell_vacuoles.columns:
                valid_distances = cell_vacuoles["distance_to_nucleus"].dropna()
                if len(valid_distances) > 0:
                    mean_distance_to_nucleus = valid_distances.mean()
            
            cell_summary.append({
                "cell_id": cell_id,
                "has_vacuole": len(cell_vacuoles) > 0,
                "num_vacuoles": len(cell_vacuoles),
                "vacuole_ids": list(cell_vacuoles["vacuole_id"]) if not cell_vacuoles.empty else [],
                "cell_area": cell_area,
                "total_vacuole_area": total_vacuole_area,
                "vacuole_area_ratio": total_vacuole_area / cell_area if cell_area > 0 else 0,
                "total_nuclei_in_vacuoles": cell_vacuoles["nuclei_count"].sum() if not cell_vacuoles.empty else 0,
                "multinucleated_vacuole_count": len(cell_vacuoles[cell_vacuoles["nuclei_count"] > 1]) if not cell_vacuoles.empty else 0,
                "mean_distance_to_nucleus": mean_distance_to_nucleus  # Added mean distance to nucleus
            })
    
    else:
        # Handle the case where no vacuoles were detected
        print("Warning: No vacuoles were found or associated with cells. Returning empty results.")
        cell_summary = []
        for cell_id in cell_ids:
            # Get cell area
            cell_mask = cell_masks == cell_id
            cell_area = np.sum(cell_mask)
            
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
                "mean_distance_to_nucleus": None
            })
        
    # Create the cell summary DataFrame
    cell_summary_df = pd.DataFrame(cell_summary)
    
    # Combine the two DataFrames into a dictionary for return
    cell_vacuole_table = {
        "cell_summary": cell_summary_df,
        "vacuole_cell_mapping": vacuole_cell_df
    }
    
    # Update the vacuole masks to only include vacuoles associated with cells
    vacuole_masks = associated_vacuoles
    
    # Calculate how many vacuoles were discarded
    total_detected = num_vacuoles
    total_kept = len(vacuole_cell_mapping)
    print(f"Kept {total_kept} out of {total_detected} detected vacuoles ({total_kept/total_detected*100:.1f}%)")
    print(f"Discarded {total_detected - total_kept} vacuoles that didn't sufficiently overlap with cells")
    
    # Process cytoplasm masks if provided
    updated_cytoplasm_masks = None
    if cytoplasm_masks is not None:
        # Make a copy of the cytoplasm masks
        updated_cytoplasm_masks = cytoplasm_masks.copy()
        
        # For each vacuole that was successfully mapped to a cell
        for _, row in vacuole_cell_df.iterrows():
            vacuole_id = row['vacuole_id']
            cell_id = row['cell_id']
            
            # Get the vacuole mask
            vacuole_mask = vacuole_masks == vacuole_id
            
            # Remove the vacuole region from the corresponding cytoplasm
            cytoplasm_mask = updated_cytoplasm_masks == cell_id
            updated_cytoplasm_masks[cytoplasm_mask & vacuole_mask] = 0
        
        print(f"Updated cytoplasm masks by removing {len(vacuole_cell_df)} vacuole regions")
    
    # Return the appropriate tuple based on whether cytoplasm_masks was provided
    if updated_cytoplasm_masks is not None:
        return vacuole_masks, cell_vacuole_table, updated_cytoplasm_masks
    else:
        return vacuole_masks, cell_vacuole_table
    

def create_vacuole_boundary_visualization(
    image,
    vacuole_channel_index,
    cell_masks,
    vacuole_masks,
    vacuole_cell_mapping=None,
    channel_names=None,
    channel_cmaps=None,
    show_nuclei_peaks=True
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
                temp_img[:,:,c] = enhanced_img[c] / (enhanced_img[c].max() if enhanced_img[c].max() > 0 else 1.0)
            
            cell_boundary_img = mark_boundaries(
                temp_img, 
                cell_masks, 
                color=(0, 1, 0),  # Green for cells
                mode='thick'
            )
            
            # Update the green channel with cell boundaries - make them more prominent
            cell_boundary_intensity = 1.2 * enhanced_img[1].max()  # Increase intensity by 20%
            enhanced_img[1] = np.maximum(enhanced_img[1], cell_boundary_img[:,:,1] * cell_boundary_intensity)
            # Cap values at 1.0 if normalized
            if enhanced_img.dtype == np.float32 or enhanced_img.dtype == np.float64:
                enhanced_img[1] = np.minimum(enhanced_img[1], 1.0 if enhanced_img[1].max() <= 1.0 else enhanced_img[1].max())
        else:
            # For single channel image, directly add boundaries to green channel
            cell_boundary = mark_boundaries(
                base_image,
                cell_masks,
                color=(0, 1, 0),  # Green for cells
                mode='thick'
            )
            enhanced_img[1] = np.maximum(enhanced_img[1], cell_boundary[:,:,1])
        
        # Add vacuole boundaries (magenta: red + blue)
        if base_is_multichannel:
            # For multichannel image, create temporary RGB again
            vacuole_boundary_img = mark_boundaries(
                temp_img,
                vacuole_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for vacuoles
                mode='thick'
            )
            
            # Update red and blue channels with vacuole boundaries
            enhanced_img[0] = np.maximum(enhanced_img[0], vacuole_boundary_img[:,:,0] * enhanced_img[0].max())
            if num_channels > 2:  # Make sure we have a blue channel
                enhanced_img[2] = np.maximum(enhanced_img[2], vacuole_boundary_img[:,:,2] * enhanced_img[2].max())
        else:
            # For single channel, add boundaries to red and blue channels
            vacuole_boundary = mark_boundaries(
                base_image,
                vacuole_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for vacuoles
                mode='thick'
            )
            enhanced_img[0] = np.maximum(enhanced_img[0], vacuole_boundary[:,:,0])
            enhanced_img[2] = np.maximum(enhanced_img[2], vacuole_boundary[:,:,2])
        
        # Add nuclei/peak markers if requested and available
        if show_nuclei_peaks and vacuole_cell_mapping is not None and 'peak_coordinates' in vacuole_cell_mapping.columns:
            for _, row in vacuole_cell_mapping.iterrows():
                if 'peak_coordinates' in row and row['peak_coordinates']:
                    for coord in row['peak_coordinates']:
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
                                            max_val = enhanced_img[:, ny, nx].max() if base_is_multichannel else 1.0
                                            enhanced_img[0, ny, nx] = max_val  # Red
                                            enhanced_img[1, ny, nx] = max_val  # Green
                                            if num_channels > 2:  # Make sure we have a blue channel
                                                enhanced_img[2, ny, nx] = 0.0  # No blue
        
        return enhanced_img
    
    # Create merged microimage with boundaries and peaks
    merged_with_boundaries = add_boundaries_and_peaks(merged_img)
    merged_microimage = Microimage(
        merged_with_boundaries,
        channel_names="Merged",
        cmaps=channel_cmaps
    )
    
    # Create vacuole channel microimage with boundaries and peaks
    # Convert single channel to 3D for processing
    vacuole_3d = add_boundaries_and_peaks(vacuole_img, base_is_multichannel=False)
    boundaries_microimage = Microimage(
        vacuole_3d,
        channel_names=f"{channel_name}",
        cmaps=['pure_red', 'pure_green', 'pure_blue']
    )
    
    # Create the micropanel
    microimages = [merged_microimage, boundaries_microimage]
    panel = create_micropanel(microimages, add_channel_label=True)
    
    return panel