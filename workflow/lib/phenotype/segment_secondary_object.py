"""Segment secondary objects using thresholding or ML methods and visualize results.

This module provides functions for segmenting and visualizing secondary objects in microscopy images.
Both traditional threshold-based and machine learning (ML) segmentation methods are supported,
with a shared post-processing pipeline that ensures consistent output formats.

Architecture:
    - segment_second_objs(): Basic segmentation (thresholding + declumping)
    - segment_second_objs_ml(): ML-based segmentation template (Cellpose, StarDist, etc.)
    - _postprocess_secondary_objects(): Shared post-processing for both methods
        * Size filtering (Feret diameter or area)
        * Cell association (spatial overlap)
        * Cell summary statistics
        * Cytoplasm mask updates

Implementing Custom ML Segmentation:
    Users implementing segment_second_objs_ml() only need to:
    1. Extract the target channel from the image
    2. Run their ML model to get a labeled mask (e.g., Cellpose, StarDist)
    3. Return the labeled mask to the shared post-processing pipeline

Key Functions:
    - segment_second_objs(): Basic segmentation
    - segment_second_objs_ml(): ML-based segmentation template (user implements)
    - _postprocess_secondary_objects(): Shared post-processing pipeline
    - create_second_obj_boundary_visualization(): Visualize segmentation results
    - create_second_obj_standard_visualization(): Standard visualization panel

Helper Functions:
    - apply_threshold_method(): Thresholding methods (Otsu, Li)
    - apply_declumping(): CellProfiler-compatible declumping
    - get_feret_diameters(): Compute Feret diameters
    - create_empty_results(): Generate empty result structures
    - get_spatial_overlap_candidates(): Spatial indexing for cell-object association

"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation, feature, util, exposure
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from microfilm.microplot import Microimage
from lib.shared.configuration_utils import create_micropanel
from lib.shared.segment_cellpose import (
    prepare_cellpose,
    create_cellpose_model,
    CELLPOSE_VERSION,
    CELLPOSE_4X,
)
import cv2

# Check if Cellpose is available (for error messaging)
try:
    import cellpose
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


def segment_second_objs_ml(
    image,
    second_obj_channel_index,
    cell_masks=None,
    cytoplasm_masks=None,
    # Post-processing parameters (shared)
    second_obj_min_size=10,
    second_obj_max_size=200,
    size_filter_method="feret",
    max_objects_per_cell=120,
    overlap_threshold=0.1,
    nuclei_centroids=None,
    max_total_objects=1000,
    # Preprocessing parameters
    logscale=True,
    # ML-specific parameters - users add more as needed
    **ml_params,
):
    """Segment secondary objects using ML models (Cellpose, StarDist, etc.).

    This function implements ML-based segmentation for secondary objects with support
    for both Cellpose and StarDist models. Users can choose the appropriate model
    based on their object morphology:
    - Cellpose: Better for irregular shapes (vacuoles, organelles with varying morphology)
    - StarDist: Better for round/star-convex objects (nuclei-like structures)

    The shared post-processing pipeline (_postprocess_secondary_objects) will handle:
    - Size filtering (Feret diameter or area)
    - Cell association (spatial overlap)
    - Cell summary statistics
    - Cytoplasm mask updates

    Parameters
    ----------
    image : ndarray
        Multichannel image data with shape [channels, height, width]
    second_obj_channel_index : int
        Index of the channel used for secondary object detection
    cell_masks : ndarray
        Cell segmentation masks with unique integers for each cell
    cytoplasm_masks : ndarray, optional
        Cytoplasm segmentation masks. If provided, secondary object
        regions will be removed from cytoplasm masks

    second_obj_min_size : float
        Minimum size for valid secondary objects
    second_obj_max_size : float
        Maximum size for valid secondary objects
    size_filter_method : str
        Size filtering method ("feret" or "area")
    max_objects_per_cell : int
        Maximum secondary objects allowed per cell
    overlap_threshold : float
        Minimum overlap ratio to associate object with cell (0.0-1.0)
    nuclei_centroids : dict, DataFrame, or None
        Cell nuclei centroids for distance calculations
    max_total_objects : int or None
        Failsafe limit on detected objects
    logscale : bool
        Apply log scaling and normalization preprocessing to the target channel
        before segmentation. This matches the preprocessing used in segment_cellpose
        and improves segmentation performance. Default is True.

    **ml_params : dict
        Additional ML model parameters. Required and optional parameters depend on ml_method:

        Common parameters:
        - second_obj_method : str (required)
            ML model to use: "cellpose" or "stardist"
        - gpu : bool (default: False)
            Whether to use GPU acceleration

        For second_obj_method="cellpose":
        - second_obj_cellpose_model : str (default: 'cyto3')
            Cellpose model type ('cyto3', 'cyto2', 'cyto', 'nuclei', etc.)
        - second_obj_diameter : float or None (default: None)
            Expected diameter of objects in pixels. If None, estimated automatically
        - second_obj_flow_threshold : float (default: 0.4)
            Flow error threshold for Cellpose segmentation
        - second_obj_cellprob_threshold : float (default: 0.0)
            Cell probability threshold for Cellpose

        For second_obj_method="stardist":
        - second_obj_stardist_model : str (default: '2D_versatile_fluo')
            StarDist pretrained model name
        - second_obj_prob_threshold : float (default: 0.5)
            Probability threshold for object detection
        - second_obj_nms_threshold : float (default: 0.4)
            Non-maximum suppression threshold

    Returns:
    -------
    tuple
        - second_obj_masks: Labeled mask of secondary objects [height, width]
        - cell_second_obj_table: Dict with 'cell_summary' and 'second_obj_cell_mapping' DataFrames
        - updated_cytoplasm_masks: Cytoplasm masks with secondary objects removed (if provided)

    Raises:
    ------
    ValueError
        If ml_method is not 'cellpose' or 'stardist', or if required packages are not installed

    Notes:
    -----
    - All post-processing is handled by _postprocess_secondary_objects()
    - Output format is guaranteed to match segment_second_objs()
    - Requires cellpose or stardist packages to be installed
    """
    # Extract and preprocess target channel
    if logscale:
        # Use prepare_cellpose for preprocessing (ensures consistency with training)
        rgb = prepare_cellpose(
            image,
            dapi_index=second_obj_channel_index,  # Dummy - will use cyto channel
            cyto_index=second_obj_channel_index,  # Target channel
            helper_index=None,
            logscale=True,
        )
        target_channel = rgb[1]  # Extract green (log scaled + normalized)
    else:
        target_channel = image[second_obj_channel_index].copy()

    # Get ML method
    ml_method = ml_params.get("second_obj_method", None)
    if ml_method is None:
        raise ValueError(
            "second_obj_method must be specified in ml_params. "
            "Valid options: 'cellpose' or 'stardist'"
        )

    gpu = ml_params.get("gpu", False)

    # Route to appropriate ML model
    if ml_method == "cellpose":
        # Cellpose parameters
        model_type = ml_params.get("second_obj_cellpose_model", "cyto3")
        diameter = ml_params.get("second_obj_diameter", None)
        flow_threshold = ml_params.get("second_obj_flow_threshold", 0.4)
        cellprob_threshold = ml_params.get("second_obj_cellprob_threshold", 0.0)

        print(
            f"Running Cellpose {model_type} model for secondary object segmentation..."
        )
        if diameter is not None:
            print(f"  Using diameter: {diameter:.1f} pixels")
        else:
            print(f"  Diameter will be estimated automatically")
        print(f"  Flow threshold: {flow_threshold}")
        print(f"  Cell probability threshold: {cellprob_threshold}")
        print(f"  GPU: {gpu}")

        # Check Cellpose availability
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is required for ML-based secondary object segmentation. "
                "Install it with: pip install cellpose"
            )

        # Initialize Cellpose model (handles version detection and validation)
        model = create_cellpose_model(model_type, gpu=gpu)

        # Run Cellpose segmentation
        # Note: CellposeModel.eval() returns 3 values (masks, flows, styles)
        # Diameter must be specified explicitly for Cellpose 4.x
        labeled_mask, flows, styles = model.eval(
            target_channel,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        print(f"Cellpose detected {len(np.unique(labeled_mask)) - 1} secondary objects")
        if diameter is not None:
            print(f"Using diameter: {diameter:.1f} pixels")

    elif ml_method == "stardist":
        # StarDist parameters
        model_type = ml_params.get("second_obj_stardist_model", "2D_versatile_fluo")
        prob_thresh = ml_params.get("second_obj_prob_threshold", 0.5)
        nms_thresh = ml_params.get("second_obj_nms_threshold", 0.4)

        print(
            f"Running StarDist {model_type} model for secondary object segmentation..."
        )
        print(f"  Probability threshold: {prob_thresh}")
        print(f"  NMS threshold: {nms_thresh}")
        print(f"  GPU: {gpu}")

        # Import StarDist
        try:
            from stardist.models import StarDist2D
        except ImportError:
            raise ImportError(
                "StarDist is required for ML-based secondary object segmentation. "
                "Install it with: pip install stardist"
            )

        # Initialize StarDist model
        model = StarDist2D.from_pretrained(model_type)

        # Run StarDist segmentation
        labeled_mask, details = model.predict_instances(
            target_channel, prob_thresh=prob_thresh, nms_thresh=nms_thresh
        )

        print(f"StarDist detected {len(np.unique(labeled_mask)) - 1} secondary objects")

    else:
        raise ValueError(
            f"Unknown ml_method: {ml_method}. Valid options: 'cellpose' or 'stardist'"
        )

    # Shared post-processing pipeline
    return _postprocess_secondary_objects(
        second_obj_masks=labeled_mask,  # ML model output
        cell_masks=cell_masks,
        cytoplasm_masks=cytoplasm_masks,
        second_obj_min_size=second_obj_min_size,
        second_obj_max_size=second_obj_max_size,
        size_filter_method=size_filter_method,
        max_objects_per_cell=max_objects_per_cell,
        overlap_threshold=overlap_threshold,
        nuclei_centroids=nuclei_centroids,
        max_total_objects=max_total_objects,
        image=image,
        second_obj_channel_index=second_obj_channel_index,
    )


def estimate_second_obj_diameter(
    image, second_obj_channel_index, method="cellpose", **kwargs
):
    """Estimate the diameter of secondary objects in an image channel.

    This is a convenience function to help users estimate appropriate diameter
    parameters for ML-based secondary object segmentation.

    Parameters
    ----------
    image : ndarray
        Multichannel image data with shape [channels, height, width]
    second_obj_channel_index : int
        Index of the channel containing secondary objects
    method : str
        Method to use for diameter estimation:
        - "cellpose": Use Cellpose's built-in diameter estimation (default)
        - "manual": Manually measure from image statistics
    **kwargs : dict
        Additional parameters for the estimation method:
        - For method="cellpose":
            - model_type : str (default: 'cyto3')
            - gpu : bool (default: False)

    Returns:
    -------
    diameter : float
        Estimated diameter in pixels

    Examples:
    --------
    >>> diameter = estimate_second_obj_diameter(
    ...     aligned_image,
    ...     second_obj_channel_index=7,
    ...     method="cellpose",
    ...     model_type="cyto3"
    ... )
    >>> print(f"Estimated diameter: {diameter:.1f} pixels")
    """
    target_channel = image[second_obj_channel_index]

    if method == "cellpose":
        # Check Cellpose availability
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is required for diameter estimation. "
                "Install it with: pip install cellpose"
            )

        # Cellpose 4.x does not support automatic diameter estimation
        if CELLPOSE_4X:
            raise NotImplementedError(
                "Automatic diameter estimation is not supported with Cellpose 4.x. "
                "Please specify second_obj_diameter explicitly in your config, "
                "or use method='manual' for threshold-based estimation, "
                "or downgrade to Cellpose 3.x: pip install cellpose==3.1.0"
            )

        model_type = kwargs.get("model_type", "cyto3")
        gpu = kwargs.get("gpu", False)

        print(f"Estimating secondary object diameter using Cellpose {model_type}...")

        # Cellpose 3.x: Use the old API which supports diameter estimation
        from cellpose import models as cellpose_models

        model = cellpose_models.Cellpose(gpu=gpu, model_type=model_type)

        # Run segmentation with automatic diameter estimation
        _, _, _, diameter = model.eval(
            target_channel,
            diameter=None,  # Auto-estimate
            channels=[0, 0],
        )

        print(f"Estimated diameter: {diameter:.1f} pixels")
        return float(diameter)

    elif method == "manual":
        # Simple estimation based on image statistics
        # Threshold the image and measure typical object sizes
        from skimage import filters, measure
        from scipy import ndimage

        # Apply Otsu threshold
        thresh = filters.threshold_otsu(target_channel)
        binary = target_channel > thresh

        # Label objects
        labeled, _ = ndimage.label(binary)
        regions = measure.regionprops(labeled)

        if len(regions) == 0:
            print("No objects detected for diameter estimation")
            return None

        # Calculate median equivalent diameter
        diameters = [r.equivalent_diameter for r in regions]
        diameter = np.median(diameters)

        print(
            f"Estimated diameter (median of {len(regions)} objects): {diameter:.1f} pixels"
        )
        return float(diameter)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'cellpose' or 'manual'")


def apply_threshold_method(image, method="otsu_two_peak"):
    """Apply specified thresholding method to an image.

    Parameters
    ----------
    image : ndarray
        Input image (should be preprocessed with log transform and smoothing)
    method : str
        Thresholding method to use.
        Options:
        - 'otsu_two_peak': Standard Otsu thresholding (2-class)
        - 'otsu_three_peak_mid_bg': 3-class Otsu, middle class as background
        - 'otsu_three_peak_mid_fg': 3-class Otsu, middle class as foreground
        - 'min_cross_entropy': Minimum cross entropy (Li) thresholding

    Returns:
    -------
    threshold : float
        Computed threshold value
    binary_mask : ndarray
        Binary mask after thresholding
    """
    if method == "otsu_two_peak":
        # Standard two-class Otsu
        threshold = filters.threshold_otsu(image)
        binary_mask = image > threshold

    elif method == "otsu_three_peak_mid_bg":
        # Three-class Otsu, treat middle intensity class as background
        threshold = filters.threshold_multiotsu(image, classes=3)
        # Keep only the highest intensity class (threshold[1] separates mid from high)
        binary_mask = image > threshold[1]

    elif method == "otsu_three_peak_mid_fg":
        # Three-class Otsu, treat middle intensity class as foreground
        threshold = filters.threshold_multiotsu(image, classes=3)
        # Keep both middle and high intensity classes (threshold[0] separates low from mid)
        binary_mask = image > threshold[0]

    elif method == "min_cross_entropy":
        # Minimum cross entropy (Li) method
        threshold = filters.threshold_li(image)
        binary_mask = image > threshold

    else:
        raise ValueError(
            f"Unknown threshold method: {method}. "
            f"Valid options: 'otsu_two_peak', 'otsu_three_peak_mid_bg', "
            f"'otsu_three_peak_mid_fg', 'min_cross_entropy'"
        )

    return threshold, binary_mask


def segment_second_objs(
    image,
    second_obj_channel_index,
    cell_masks=None,
    cytoplasm_masks=None,
    # Size filtering
    second_obj_min_size=10,
    second_obj_max_size=200,
    size_filter_method="feret",
    # Pre-processing
    threshold_smoothing_scale=1.3488,
    threshold_method="otsu_two_peak",
    use_morphological_opening=True,
    opening_disk_radius=1,
    fill_holes="both",
    # Declumping method (CellProfiler standard)
    declump_method="shape",
    declump_mode="watershed",
    # Seed detection (CellProfiler naming)
    suppress_local_maxima=20,
    maxima_reduction_factor=None,
    # Shape-based refinement (independent from declump_method)
    use_shape_refinement=False,
    proportion_threshold=0.4,
    # Cell association
    max_objects_per_cell=120,
    overlap_threshold=0.1,
    nuclei_centroids=None,
    # Failsafe
    max_total_objects=1000,
    # Debugging
    return_threshold_output=False,
):
    """Segment secondary objects within cells using CellProfiler-compatible thresholding and declumping.

    Args:
        image (numpy.ndarray): Multichannel image data with shape [channels, height, width].
        second_obj_channel_index (int): Index of the channel used for secondary object detection.
        cell_masks (numpy.ndarray): Cell segmentation masks with unique integers for each cell.
        cytoplasm_masks (numpy.ndarray, optional): Cytoplasm segmentation masks with unique integers.
            If provided, secondary object regions will be removed from cytoplasm masks.

        second_obj_min_size (float, optional): Minimum size for valid secondary objects (default: 10).
            Interpreted as Feret diameter or area depending on size_filter_method.
        second_obj_max_size (float, optional): Maximum size for valid secondary objects (default: 200).
        size_filter_method (str, optional): Size filtering method (default: "feret").
            - "feret": Use Feret diameters (min and max widths of rotated bounding box)
            - "area": Use pixel area (CellProfiler standard)

        threshold_smoothing_scale (float, optional): Sigma for Gaussian smoothing before thresholding. Default is 1.3488.
        threshold_method (str, optional): Thresholding method to use (default: "otsu_two_peak").
            Options:
            - "otsu_two_peak": Standard 2-class Otsu thresholding
            - "otsu_three_peak_mid_bg": 3-class Otsu, keeps only highest intensity class
            - "otsu_three_peak_mid_fg": 3-class Otsu, keeps middle and high intensity classes
            - "min_cross_entropy": Minimum cross entropy (Li) thresholding
        use_morphological_opening (bool, optional): Apply opening to separate weakly connected objects (default: True).
        opening_disk_radius (int, optional): Radius of disk structuring element for opening (default: 1).
        fill_holes (str, optional): When to fill holes in segmented objects (default: "both").
            Options:
            - "threshold": Fill holes only after thresholding (before declumping)
            - "declump": Fill holes only after declumping (per-label filling)
            - "both": Fill holes after both thresholding and declumping
            - "none": Do not fill holes at any stage

        declump_method (str, optional): Method for separating clumped objects (default: "shape").
            CellProfiler standard methods:
            - "none": No declumping (connected components only)
            - "shape": Distance transform peaks (radial distance)
            - "intensity": Local intensity maxima
            - "shape_intensity": Combined distance + intensity peaks

        declump_mode (str, optional): Watershed segmentation mode (default: "watershed").
            - "watershed": Standard watershed from markers
            - "propagate": Distance propagation variant
            - "none": Use markers only without watershed

        suppress_local_maxima (int, optional): Minimum spacing between seed points in pixels (default: 20).
            CellProfiler parameter. Controls spatial separation of detected peaks.

        maxima_reduction_factor (float or None, optional): H-minima threshold for suppressing weak peaks (default: None).
            Range: 0.0-1.0. Higher values = more aggressive suppression.
            If None, no h-minima filtering applied.
            Formula: h = maxima_reduction_factor * (peak_map_max - peak_map_min)
            This is applied DURING seed detection (before watershed).

        use_shape_refinement (bool, optional): Apply boundary/perimeter quality control after declumping (default: False).
            Custom feature not in CellProfiler. When enabled, evaluates watershed splits
            and rejects splits where the dividing boundary is long relative to perimeter.
            This is applied AFTER watershed declumping as a refinement step.

        proportion_threshold (float, optional): Boundary/perimeter ratio threshold for shape refinement (default: 0.4).
            Only used when use_shape_refinement=True.
            Splits accepted if boundary_length / perimeter < proportion_threshold.

        max_objects_per_cell (int, optional): Maximum secondary objects allowed per cell (default: 120).
        overlap_threshold (float, optional): Minimum overlap ratio to associate object with cell (default: 0.1).
        nuclei_centroids (dict or DataFrame, optional): Cell nuclei centroids for distance calculations.
            Format: {nuclei_id: (i, j)} or DataFrame with columns 'i', 'j'.

        max_total_objects (int or None, optional): Failsafe limit on detected objects (default: 1000).
            Returns empty results if exceeded to avoid processing over-segmented images.

        return_threshold_output (bool, optional): If True, returns intermediate thresholding results
            for debugging and visualization (default: False). When enabled, the threshold_output
            dictionary contains:
            - 'binary_mask': Binary mask after thresholding, hole filling, and opening (before declumping)
            - 'threshold_value': Computed threshold value from apply_threshold_method()
            - 'preprocessed_channel': Log-transformed and Gaussian-smoothed channel used for thresholding

    Returns:
        tuple: Returns depend on return_threshold_output flag:

        If return_threshold_output=False (default):
            - second_obj_masks (numpy.ndarray): Labeled mask of secondary objects
            - cell_second_obj_table (dict): Dictionary with DataFrames containing associations
            - updated_cytoplasm_masks (numpy.ndarray): Updated cytoplasm masks (if cytoplasm_masks provided)

        If return_threshold_output=True:
            - second_obj_masks (numpy.ndarray): Labeled mask of secondary objects
            - cell_second_obj_table (dict): Dictionary with DataFrames containing associations
            - updated_cytoplasm_masks (numpy.ndarray): Updated cytoplasm masks (if cytoplasm_masks provided)
            - threshold_output (dict): Intermediate thresholding results with keys:
                - 'binary_mask': Binary mask before declumping
                - 'threshold_value': Threshold value used
                - 'preprocessed_channel': Preprocessed channel image
    """
    # Extract the secondary object channel
    second_obj_img = image[second_obj_channel_index]
    second_obj_img = np.clip(second_obj_img, a_min=0, a_max=None)

    # Apply log transform and smoothing
    second_obj_log = exposure.adjust_log(second_obj_img + 1)
    second_obj_smooth = filters.gaussian(
        second_obj_log, sigma=threshold_smoothing_scale
    )

    # Apply selected threshold method
    thresh, binary_mask = apply_threshold_method(
        second_obj_smooth, method=threshold_method
    )

    # Fill holes after thresholding (if enabled)
    if fill_holes in ["threshold", "both"]:
        binary_mask = ndimage.binary_fill_holes(binary_mask)

    # Early exit if no objects found
    if not np.any(binary_mask):
        print("No objects detected after thresholding")
        empty_results = create_empty_results(
            cell_masks, cytoplasm_masks, nuclei_centroids
        )

        # Handle return_threshold_output for empty case
        if return_threshold_output:
            threshold_output = {
                "binary_mask": binary_mask,
                "threshold_value": thresh,
                "preprocessed_channel": second_obj_smooth,
            }
            # Add threshold_output to the tuple
            if cytoplasm_masks is not None:
                return (*empty_results, threshold_output)
            else:
                return (*empty_results, threshold_output)
        else:
            return empty_results

    # Failsafe: Check for excessive objects early
    if max_total_objects is not None:
        temp_labeled, num_components = ndimage.label(binary_mask)
        if num_components > max_total_objects:
            print(
                f"Failsafe triggered: Detected {num_components} objects (limit: {max_total_objects})"
            )
            print("Returning empty results to avoid processing over-segmented image")
            empty_results = create_empty_results(
                cell_masks, cytoplasm_masks, nuclei_centroids
            )

            # Handle return_threshold_output for failsafe case
            if return_threshold_output:
                threshold_output = {
                    "binary_mask": binary_mask,
                    "threshold_value": thresh,
                    "preprocessed_channel": second_obj_smooth,
                }
                if cytoplasm_masks is not None:
                    return (*empty_results, threshold_output)
                else:
                    return (*empty_results, threshold_output)
            else:
                return empty_results

    # Morphological opening
    if use_morphological_opening:
        binary_mask = apply_morphological_opening(
            binary_mask, opening_disk_radius=opening_disk_radius
        )

    # Capture intermediate state for visualization
    if return_threshold_output:
        threshold_binary_mask = binary_mask.copy()
        threshold_value_stored = thresh
        threshold_preprocessed_channel = second_obj_smooth.copy()

    # Declumping
    declumped = apply_declumping(
        binary_mask,
        second_obj_smooth,
        declump_method=declump_method,
        declump_mode=declump_mode,
        suppress_local_maxima=suppress_local_maxima,
        maxima_reduction_factor=maxima_reduction_factor,
    )

    print(
        f"After declumping ({declump_method}): {len(np.unique(declumped)) - 1} objects"
    )

    # Optionally apply shape-based refinement (independent from declump_method)
    if use_shape_refinement:
        print("Applying shape-based boundary/perimeter refinement...")
        declumped = shape_based_declumping(
            declumped > 0,
            second_obj_img=second_obj_img,
            min_distance=suppress_local_maxima,
            proportion_threshold=proportion_threshold,
        )
        print(f"After shape refinement: {len(np.unique(declumped)) - 1} objects")

    # Fill holes after declumping (if enabled)
    if fill_holes in ["declump", "both"]:
        unique_labels = np.unique(declumped[declumped > 0])
        for label in unique_labels:
            mask = declumped == label
            filled = ndimage.binary_fill_holes(mask)
            declumped[filled] = label

    # Apply shared post-processing pipeline: size filtering, cell association, statistics, cytoplasm updates
    post_results = _postprocess_secondary_objects(
        second_obj_masks=declumped,
        cell_masks=cell_masks,
        cytoplasm_masks=cytoplasm_masks,
        second_obj_min_size=second_obj_min_size,
        second_obj_max_size=second_obj_max_size,
        size_filter_method=size_filter_method,
        max_objects_per_cell=max_objects_per_cell,
        overlap_threshold=overlap_threshold,
        nuclei_centroids=nuclei_centroids,
        max_total_objects=max_total_objects,
        image=image,
        second_obj_channel_index=second_obj_channel_index,
    )

    # Handle return_threshold_output flag for debugging
    if return_threshold_output:
        # Create threshold output dictionary
        threshold_output = {
            "binary_mask": threshold_binary_mask,
            "threshold_value": threshold_value_stored,
            "preprocessed_channel": threshold_preprocessed_channel,
        }
        # Append threshold_output to results tuple
        return (*post_results, threshold_output)
    else:
        # Standard return (backward compatible)
        return post_results


def create_second_obj_boundary_visualization(
    image,
    second_obj_channel_index,
    cell_masks,
    second_obj_masks,
    channel_names=None,
    channel_cmaps=None,
):
    """Create enhanced visualization showing cells and secondary objects.

    Args:
        image (numpy.ndarray): Multichannel image data with shape [channels, height, width].
        second_obj_channel_index (int): Index of the channel used for secondary object detection.
        cell_masks (numpy.ndarray): Cell segmentation masks with unique integers for each cell.
        second_obj_masks (numpy.ndarray): Secondary object segmentation masks with original secondary object IDs.
        channel_names (list of str, optional): Names for each channel in the image.
        channel_cmaps (list of str, optional): Color maps for each channel in the image.

    Returns:
        matplotlib.figure.Figure: The created micropanel figure showing the cell boundaries (green)
            and secondary object boundaries (magenta) overlaid on the image.
    """
    if channel_names is None or len(channel_names) <= second_obj_channel_index:
        channel_name = f"Channel {second_obj_channel_index}"
    else:
        channel_name = channel_names[second_obj_channel_index]

    # Get secondary object channel
    second_obj_img = image[second_obj_channel_index].copy()

    # Create a copy of the original image for the merged view with boundaries
    merged_img = image.copy()

    # Function to add boundaries to an image
    def add_boundaries(base_image, base_is_multichannel=True):
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

        # Add secondary object boundaries (magenta: red + blue)
        if base_is_multichannel:
            # For multichannel image, create temporary RGB again
            second_obj_boundary_img = mark_boundaries(
                temp_img,
                second_obj_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for secondary objects
                mode="thick",
            )

            # Update red and blue channels with secondary object boundaries
            enhanced_img[0] = np.maximum(
                enhanced_img[0],
                second_obj_boundary_img[:, :, 0] * enhanced_img[0].max(),
            )
            if num_channels > 2:  # Make sure we have a blue channel
                enhanced_img[2] = np.maximum(
                    enhanced_img[2],
                    second_obj_boundary_img[:, :, 2] * enhanced_img[2].max(),
                )
        else:
            # For single channel, add boundaries to red and blue channels
            second_obj_boundary = mark_boundaries(
                base_image,
                second_obj_masks > 0,  # Binary mask
                color=(1, 0, 1),  # Magenta for secondary objects
                mode="thick",
            )
            enhanced_img[0] = np.maximum(enhanced_img[0], second_obj_boundary[:, :, 0])
            enhanced_img[2] = np.maximum(enhanced_img[2], second_obj_boundary[:, :, 2])

        return enhanced_img

    # Create merged microimage with boundaries
    merged_with_boundaries = add_boundaries(merged_img)
    merged_microimage = Microimage(
        merged_with_boundaries, channel_names="Merged", cmaps=channel_cmaps
    )

    # Create secondary object channel microimage with boundaries
    # Convert single channel to 3D for processing
    second_obj_3d = add_boundaries(second_obj_img, base_is_multichannel=False)
    boundaries_microimage = Microimage(
        second_obj_3d,
        channel_names=f"{channel_name}",
        cmaps=["pure_red", "pure_green", "pure_blue"],
    )

    # Create the micropanel
    microimages = [merged_microimage, boundaries_microimage]
    panel = create_micropanel(microimages, add_channel_label=True)

    return panel


def create_second_obj_standard_visualization(
    aligned_image,
    second_obj_channel_index,
    second_obj_channel_name,
    second_obj_masks,
    threshold_output=None,
    label_color="magenta",
):
    """Create standard visualization panel for secondary object segmentation.

    Parameters
    ----------
    aligned_image : ndarray
        Multichannel aligned image [channels, height, width]
    second_obj_channel_index : int
        Index of the channel used for secondary object detection
    second_obj_channel_name : str
        Name of the secondary object channel (e.g., "CDPK1")
    second_obj_masks : ndarray
        Labeled mask of segmented secondary objects
    threshold_output : dict, optional
        Dictionary containing threshold debugging output with keys:
        - 'preprocessed_channel': Log-transformed and Gaussian-smoothed channel
        - 'binary_mask': Binary mask after thresholding
        If None, creates simple 1x2 panel. If provided, creates 2x2 panel.
    label_color : str, optional
        Color for channel labels (default: 'magenta')

    Returns:
    -------
    panel : Micropanel
        Micropanel object with visualizations
    """
    from lib.shared.configuration_utils import random_cmap

    # Build secondary object colormap
    second_obj_cmap = random_cmap(num_colors=len(np.unique(second_obj_masks)))

    if threshold_output:
        # 2x2 grid when threshold_output is provided
        micro_images = [
            Microimage(
                threshold_output["preprocessed_channel"],
                channel_names="Preprocessed",
                cmaps="gray",
            ),
            Microimage(
                threshold_output["binary_mask"],
                channel_names="Threshold Binary",
                cmaps="gray",
            ),
            Microimage(
                aligned_image[second_obj_channel_index],
                channel_names=f"{second_obj_channel_name} (Raw)",
                cmaps="gray",
            ),
            Microimage(
                second_obj_masks,
                cmaps=second_obj_cmap,
                channel_names="Secondary Objects",
            ),
        ]
        num_cols = 2
    else:
        # 1x2 grid when threshold_output is None
        micro_images = [
            Microimage(
                aligned_image[second_obj_channel_index],
                channel_names=f"{second_obj_channel_name} (Raw)",
                cmaps="gray",
            ),
            Microimage(
                second_obj_masks,
                cmaps=second_obj_cmap,
                channel_names="Secondary Objects",
            ),
        ]
        num_cols = 2

    panel = create_micropanel(
        micro_images,
        add_channel_label=True,
        num_cols=num_cols,
    )

    # Set all channel labels to specified color
    for ax in panel.fig.axes:
        for text in ax.texts:
            text.set_color(label_color)

    return panel


def get_feret_diameters(coords):
    """Compute the minimum and maximum Feret diameters of a 2D shape.

    The Feret diameters are calculated using OpenCV's minAreaRect, which finds
    the smallest-area rotated bounding rectangle that encloses the input coordinates.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        An array of (x, y) coordinates representing the pixels or contour of a region.

    Returns:
    -------
    feret_min : float
        The shortest distance between two parallel lines tangent to the object
        (i.e., the minimum Feret diameter).

    feret_max : float
        The longest distance between two parallel lines tangent to the object
        (i.e., the maximum Feret diameter).

    Notes:
    -----
    - This method assumes the input coordinates define a planar shape (e.g., from a binary mask or regionprops).
    - The returned values are in the same units as the input coordinates (typically pixels).
    - Internally uses OpenCV's cv2.minAreaRect for fast and robust measurement.
    """
    cnt = coords.astype(np.int32)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    return min(w, h), max(w, h)


def get_feret_diameters(coords):
    """Compute the minimum and maximum Feret diameters of a 2D shape.

    The Feret diameters are calculated using OpenCV's minAreaRect, which finds
    the smallest-area rotated bounding rectangle that encloses the input coordinates.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        An array of (x, y) coordinates representing the pixels or contour of a region.

    Returns:
    -------
    feret_min : float
        The shortest distance between two parallel lines tangent to the object
        (i.e., the minimum Feret diameter).

    feret_max : float
        The longest distance between two parallel lines tangent to the object
        (i.e., the maximum Feret diameter).

    Notes:
    -----
    - This method assumes the input coordinates define a planar shape (e.g., from a binary mask or regionprops).
    - The returned values are in the same units as the input coordinates (typically pixels).
    - Internally uses OpenCV's cv2.minAreaRect for fast and robust measurement.
    """
    cnt = coords.astype(np.int32)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    return min(w, h), max(w, h)


def apply_morphological_opening(binary_mask, opening_disk_radius=1):
    """Apply morphological opening to separate weakly connected secondary objects.

    Parameters
    ----------
    binary_mask : ndarray
        Binary mask of secondary objects
    opening_disk_radius : int
        Radius of disk structuring element (larger = more aggressive)

    Returns:
    -------
    opened_mask : ndarray
        Morphologically opened mask
    """
    footprint = morphology.disk(max(1, opening_disk_radius))
    opened = morphology.binary_opening(binary_mask, footprint=footprint)

    # Recover small objects that were removed by opening
    removed = binary_mask & ~opened
    small_objects, num = ndimage.label(removed)

    # Only recover objects at least as large as the structuring element
    min_recoverable_size = np.pi * opening_disk_radius**2
    for i in range(1, num + 1):
        obj_mask = small_objects == i
        if np.sum(obj_mask) >= min_recoverable_size:
            opened |= obj_mask

    return opened


def apply_h_minima_suppression(peak_map, h_factor):
    """Apply h-minima transform to suppress weak local maxima.

    This complements spatial suppression (min_distance) by filtering peaks
    based on their prominence/height in the distance or intensity map.

    Parameters
    ----------
    peak_map : ndarray
        Distance transform or intensity image
    h_factor : float
        Height threshold factor (0.0-1.0)
        h = h_factor * (peak_map.max() - peak_map.min())
        Higher values = more aggressive suppression

    Returns:
    -------
    filtered_map : ndarray
        Map with weak maxima suppressed
    """
    if h_factor <= 0 or h_factor > 1:
        raise ValueError(f"h_factor must be in (0, 1], got {h_factor}")

    # Calculate absolute height threshold
    h = h_factor * (peak_map.max() - peak_map.min())

    # Apply h-minima transform
    filtered_map = morphology.h_minima(peak_map, h=h)

    return filtered_map


def apply_declumping(
    binary_mask,
    second_obj_smooth,
    declump_method,
    declump_mode,
    suppress_local_maxima,
    maxima_reduction_factor,
):
    """Apply declumping based on CellProfiler-compatible method selection.

    Parameters
    ----------
    binary_mask : ndarray
        Binary mask of secondary objects
    second_obj_smooth : ndarray
        Smoothed intensity image (log + Gaussian filtered)
    declump_method : str
        "none", "shape", "intensity", "shape_intensity", "distance"
    declump_mode : str
        "watershed", "propagate", "none"
    suppress_local_maxima : int
        Minimum distance between peaks (spatial constraint)
    maxima_reduction_factor : float or None
        H-minima threshold (0.0-1.0), None=disabled

    Returns:
    -------
    declumped : ndarray
        Labeled mask after declumping

    Notes:
    -----
    Shape refinement is NOT handled here - it's applied as optional refinement
    after this function in the main pipeline.
    """
    # Method 1: No declumping
    if declump_method == "none":
        declumped, _ = ndimage.label(binary_mask)
        return declumped

    # Method 2: Shape-based (distance transform)
    if declump_method in ["shape", "distance"]:
        peak_map = ndimage.distance_transform_edt(binary_mask)

    # Method 3: Intensity-based
    elif declump_method == "intensity":
        # Use smoothed intensity within mask
        peak_map = second_obj_smooth.copy()
        peak_map[~binary_mask] = 0

    # Method 4: Combined shape + intensity
    elif declump_method == "shape_intensity":
        # Normalize both maps to [0, 1] and average
        distance_map = ndimage.distance_transform_edt(binary_mask)
        distance_norm = distance_map / (distance_map.max() + 1e-10)

        intensity_map = second_obj_smooth.copy()
        intensity_map[~binary_mask] = 0
        intensity_norm = intensity_map / (intensity_map.max() + 1e-10)

        peak_map = (distance_norm + intensity_norm) / 2

    else:
        raise ValueError(f"Unknown declump_method: {declump_method}")

    # Apply h-minima suppression if requested
    if maxima_reduction_factor is not None:
        peak_map = apply_h_minima_suppression(peak_map, maxima_reduction_factor)

    # Detect local maxima
    local_max = feature.peak_local_max(
        peak_map,
        min_distance=suppress_local_maxima,
        labels=binary_mask,
        exclude_border=False,
    )

    # Create markers
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(local_max) == 0:
        # No peaks found, return connected components
        declumped, _ = ndimage.label(binary_mask)
        return declumped

    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)

    # Apply declump_mode
    if declump_mode == "none":
        # Use markers only (no watershed)
        declumped = markers.copy()

    elif declump_mode == "watershed":
        # Standard watershed with negative distance
        if declump_method in ["shape", "shape_intensity"]:
            # Use distance transform for watershed
            distance = ndimage.distance_transform_edt(binary_mask)
            declumped = segmentation.watershed(-distance, markers, mask=binary_mask)
        else:
            # For pure intensity, watershed on negative intensity
            intensity = second_obj_smooth.copy()
            intensity[~binary_mask] = intensity.max()
            declumped = segmentation.watershed(intensity, markers, mask=binary_mask)

    elif declump_mode == "propagate":
        # Propagate from seeds using positive distance
        distance = ndimage.distance_transform_edt(binary_mask)
        declumped = segmentation.watershed(distance, markers, mask=binary_mask)

    else:
        raise ValueError(f"Unknown declump_mode: {declump_mode}")

    # Recover unassigned regions
    missing = (declumped == 0) & binary_mask
    if np.any(missing):
        labeled_missing, _ = ndimage.label(missing)
        if declumped.max() > 0:
            labeled_missing[labeled_missing > 0] += declumped.max()
        declumped += labeled_missing

    return declumped


def shape_based_declumping(
    binary_mask, second_obj_img=None, min_distance=20, proportion_threshold=0.4
):
    """Split connected components only when the separating boundary is short relative to the region perimeter.

    Parameters
    ----------
    binary_mask : ndarray
        Input binary secondary object mask
    second_obj_img : ndarray, optional
        Intensity image (currently unused, kept for API compatibility)
    min_distance : int
        Minimum distance between peaks for watershed markers
    proportion_threshold : float
        If boundary_length / perimeter < proportion_threshold, accept the split
        Example: 0.12 means cut must be < 12% of perimeter to split

    Returns:
    -------
    labeled : ndarray
        Labeled mask after shape-based declumping
    """
    labeled_out = np.zeros_like(binary_mask, dtype=int)
    next_label = 1

    # Label connected regions
    regions_lab, n = ndimage.label(binary_mask)

    for region_label in range(1, n + 1):
        region_mask = regions_lab == region_label
        if region_mask.sum() == 0:
            continue

        # Distance transform and find peaks
        dist = ndimage.distance_transform_edt(region_mask)
        peaks = feature.peak_local_max(
            dist, min_distance=min_distance, labels=region_mask, exclude_border=False
        )

        # If only one peak, keep as single object
        if len(peaks) <= 1:
            labeled_out[region_mask] = next_label
            next_label += 1
            continue

        # Create markers and apply watershed
        markers = np.zeros_like(region_mask, dtype=int)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        local_watershed = segmentation.watershed(-dist, markers, mask=region_mask)

        # Vectorized boundary detection
        lab = local_watershed

        # Detect boundaries by comparing with neighbors
        # Vertical boundaries (compare rows)
        vertical_boundary = (
            (lab[:-1, :] != lab[1:, :]) & (lab[:-1, :] > 0) & (lab[1:, :] > 0)
        )

        # Horizontal boundaries (compare columns)
        horizontal_boundary = (
            (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] > 0) & (lab[:, 1:] > 0)
        )

        # Count total boundary pixels
        # We need to count them separately since they have different shapes
        boundary_length = np.sum(vertical_boundary) + np.sum(horizontal_boundary)

        prop = measure.regionprops(region_mask.astype(np.uint8))[0]
        perimeter = prop.perimeter if prop.perimeter > 0 else 1.0

        # Accept split if boundary is short relative to perimeter
        if (boundary_length / perimeter) < proportion_threshold:
            sublabels = np.unique(local_watershed[local_watershed > 0])
            for s in sublabels:
                labeled_out[local_watershed == s] = next_label
                next_label += 1
        else:
            # Reject split, keep as single object
            labeled_out[region_mask] = next_label
            next_label += 1

    return labeled_out


def create_empty_results(cell_masks, cytoplasm_masks, nuclei_centroids=None):
    """Helper function to create empty results when no secondary objects are found.

    Parameters
    ----------
    cell_masks : ndarray
        Cell segmentation masks
    cytoplasm_masks : ndarray, optional
        Cytoplasm segmentation masks
    nuclei_centroids : dict or DataFrame, optional
        Nuclei centroids information

    Returns:
    -------
    tuple
        Empty secondary object masks, cell_second_obj_table dict, and optionally cytoplasm_masks
    """
    cell_ids = np.unique(cell_masks[cell_masks > 0])
    empty_second_obj_masks = np.zeros_like(cell_masks)

    cell_summary = []
    for cell_id in cell_ids:
        cell_area = np.sum(cell_masks == cell_id)
        summary_entry = {
            "cell_id": cell_id,
            "has_second_obj": False,
            "num_second_objs": 0,
            "second_obj_ids": [],
            "cell_area": cell_area,
            "total_second_obj_area": 0,
            "second_obj_area_ratio": 0,
            "mean_second_obj_diameter": None,
        }

        # Add cell nucleus distance fields if nuclei_centroids was provided
        if nuclei_centroids is not None:
            summary_entry["mean_distance_to_nucleus"] = None

        cell_summary.append(summary_entry)

    cell_second_obj_table = {
        "cell_summary": pd.DataFrame(cell_summary),
        "second_obj_cell_mapping": pd.DataFrame(),
    }

    if cytoplasm_masks is not None:
        return empty_second_obj_masks, cell_second_obj_table, cytoplasm_masks
    else:
        return empty_second_obj_masks, cell_second_obj_table


def get_spatial_overlap_candidates(second_obj_regions, cell_masks):
    """Use bounding boxes to pre-filter which cells could overlap with each secondary object.

    Parameters
    ----------
    second_obj_regions : dict
        Dictionary mapping second_obj_id to regionprops
    cell_masks : ndarray
        Cell segmentation masks

    Returns:
    -------
    candidates : dict
        Dictionary mapping second_obj_id to list of candidate cell_ids
    """
    # Get all cell regions with their bounding boxes
    cell_regions = measure.regionprops(cell_masks)
    cell_bboxes = {
        r.label: r.bbox for r in cell_regions
    }  # (min_row, min_col, max_row, max_col)

    candidates = {}

    for second_obj_id, vac_region in second_obj_regions.items():
        vac_bbox = vac_region.bbox  # (min_row, min_col, max_row, max_col)

        # Find cells whose bounding boxes intersect with this secondary object's bbox
        overlapping_cells = []
        for cell_id, cell_bbox in cell_bboxes.items():
            # Check if bounding boxes overlap
            if not (
                vac_bbox[2] < cell_bbox[0]  # second_obj above cell
                or vac_bbox[0] > cell_bbox[2]  # second_obj below cell
                or vac_bbox[3] < cell_bbox[1]  # second_obj left of cell
                or vac_bbox[1] > cell_bbox[3]
            ):  # second_obj right of cell
                overlapping_cells.append(cell_id)

        candidates[second_obj_id] = overlapping_cells

    return candidates


def _postprocess_secondary_objects(
    second_obj_masks,
    cell_masks,
    cytoplasm_masks,
    second_obj_min_size,
    second_obj_max_size,
    size_filter_method,
    max_objects_per_cell,
    overlap_threshold,
    nuclei_centroids,
    max_total_objects,
    image=None,
    second_obj_channel_index=None,
):
    """Apply post-processing pipeline to secondary object masks.

    This function performs the shared post-processing steps for both
    basic thresholding and ML-based secondary object segmentation:
    1. Size filtering (Feret diameter or area)
    2. Cell association (spatial overlap)
    3. Cell summary statistics
    4. Cytoplasm mask updates

    Parameters
    ----------
    second_obj_masks : ndarray
        Labeled mask of secondary objects (integer labels, background=0)
    cell_masks : ndarray
        Cell segmentation masks with unique integers for each cell
    cytoplasm_masks : ndarray or None
        Cytoplasm segmentation masks. If provided, secondary object
        regions will be removed from cytoplasm masks
    second_obj_min_size : float
        Minimum size for valid secondary objects
    second_obj_max_size : float
        Maximum size for valid secondary objects
    size_filter_method : str
        Size filtering method ("feret" or "area")
    max_objects_per_cell : int
        Maximum secondary objects allowed per cell
    overlap_threshold : float
        Minimum overlap ratio to associate object with cell (0.0-1.0)
    nuclei_centroids : dict, DataFrame, or None
        Cell nuclei centroids for distance calculations.
        Format: {nuclei_id: (i, j)} or DataFrame with 'i', 'j' columns
    max_total_objects : int or None
        Failsafe limit on detected objects. Returns empty results if exceeded
    image : ndarray, optional
        Multichannel image [channels, height, width].
        Only needed if nuclei_centroids provided (for distance calculations)
    second_obj_channel_index : int, optional
        Index of secondary object channel.
        Only needed if nuclei_centroids provided (for distance calculations)

    Returns:
    -------
    tuple
        - second_obj_masks: Filtered and renumbered secondary object masks
        - cell_second_obj_table: Dict with 'cell_summary' and 'second_obj_cell_mapping' DataFrames
        - updated_cytoplasm_masks: Cytoplasm masks with secondary objects removed (or None)

    Notes:
    -----
    - This function is shared by both segment_second_objs() and segment_second_objs_ml()
    - Input masks should already be labeled (not binary)
    - Empty input masks are handled gracefully
    """
    # Handle empty input
    if not np.any(second_obj_masks):
        print("No objects detected in input masks")
        return create_empty_results(cell_masks, cytoplasm_masks, nuclei_centroids)

    # Failsafe: Check for excessive objects early
    num_input_objects = len(np.unique(second_obj_masks)) - 1  # Exclude background
    if max_total_objects is not None and num_input_objects > max_total_objects:
        print(
            f"Failsafe triggered: Detected {num_input_objects} objects (limit: {max_total_objects})"
        )
        print("Returning empty results to avoid processing over-segmented image")
        return create_empty_results(cell_masks, cytoplasm_masks, nuclei_centroids)

    # Filter by size
    print(f"Filtering by {size_filter_method}...")
    regions = measure.regionprops(second_obj_masks)
    valid_labels = []

    if size_filter_method == "feret":
        # Feret diameter filtering
        for region in regions:
            coords = region.coords[:, [1, 0]]  # (x, y) format
            if len(coords) < 3:
                continue

            feret_min, feret_max = get_feret_diameters(coords)
            if second_obj_min_size <= feret_min and feret_max <= second_obj_max_size:
                valid_labels.append(region.label)

    elif size_filter_method == "area":
        # Area-based filtering (CellProfiler standard)
        for region in regions:
            if second_obj_min_size <= region.area <= second_obj_max_size:
                valid_labels.append(region.label)

    else:
        raise ValueError(f"Unknown size_filter_method: {size_filter_method}")

    if not valid_labels:
        print(f"No valid secondary objects found after {size_filter_method} filtering")
        return create_empty_results(cell_masks, cytoplasm_masks, nuclei_centroids)

    print(
        f"After {size_filter_method} filtering: {len(valid_labels)} valid secondary objects"
    )

    # Create valid secondary objects mask with renumbered labels
    labeled_second_objs = np.zeros_like(second_obj_masks)
    for i, lbl in enumerate(valid_labels, start=1):
        labeled_second_objs[second_obj_masks == lbl] = i

    num_second_objs = len(valid_labels)

    # Get cell IDs
    cell_ids = np.unique(cell_masks[cell_masks > 0])

    # Prepare nuclei centroids - this is for cell nuclei distance calculations
    nuclei_centroids_dict = None
    if nuclei_centroids is not None:
        if isinstance(nuclei_centroids, pd.DataFrame):
            nuclei_centroids_dict = {
                row.get("nuclei_id", idx): (row["i"], row["j"])
                for idx, row in nuclei_centroids.iterrows()
            }
        else:
            nuclei_centroids_dict = nuclei_centroids

    # Pre-compute region properties for all secondary objects
    second_obj_regions = {
        region.label: region for region in measure.regionprops(labeled_second_objs)
    }

    # Pre-compute which cells could overlap with each secondary object
    print("Computing spatial overlap candidates...")
    overlap_candidates = get_spatial_overlap_candidates(second_obj_regions, cell_masks)

    # Initialize tracking variables
    second_obj_cell_mapping = []
    second_objs_per_cell = {cell_id: 0 for cell_id in cell_ids}

    # Process each secondary object
    print("Processing secondary object-cell associations...")
    for second_obj_id in range(1, num_second_objs + 1):
        if second_obj_id not in second_obj_regions:
            continue

        region = second_obj_regions[second_obj_id]
        second_obj_mask = labeled_second_objs == second_obj_id
        second_obj_area = region.area
        second_obj_centroid = region.centroid

        # Calculate equivalent diameter for this secondary object
        second_obj_diameter = 2 * np.sqrt(second_obj_area / np.pi)

        # Initialize mapping entry with basic info
        mapping_entry = {
            "second_obj_id": second_obj_id,
            "second_obj_area": second_obj_area,
            "second_obj_diameter": second_obj_diameter,
        }

        # Calculate distance to nearest cell nucleus
        if nuclei_centroids_dict is not None:
            min_dist = np.inf
            nearest_nucleus_id = None
            for nuc_id, nuc_centroid in nuclei_centroids_dict.items():
                dist = np.sqrt(
                    (second_obj_centroid[0] - nuc_centroid[0]) ** 2
                    + (second_obj_centroid[1] - nuc_centroid[1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_nucleus_id = nuc_id

            mapping_entry["distance_to_nucleus"] = (
                min_dist if min_dist != np.inf else None
            )
            mapping_entry["nearest_nucleus_id"] = nearest_nucleus_id

        # Find best overlapping cell
        best_cell_id = None
        best_overlap = 0

        # Check spatial overlap candidates
        candidate_cells = overlap_candidates.get(second_obj_id, [])

        for cell_id in candidate_cells:
            if second_objs_per_cell[cell_id] >= max_objects_per_cell:
                continue

            # Calculate overlap efficiently
            cell_mask = cell_masks == cell_id
            overlap = np.sum(second_obj_mask & cell_mask)

            if overlap > 0:
                overlap_ratio = overlap / second_obj_area
                if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cell_id = cell_id

        # Add successful associations
        if best_cell_id is not None:
            mapping_entry["cell_id"] = best_cell_id
            mapping_entry["overlap_ratio"] = best_overlap

            second_obj_cell_mapping.append(mapping_entry)
            second_objs_per_cell[best_cell_id] += 1

    # Create secondary object-cell mapping DataFrame
    second_obj_cell_df = pd.DataFrame(second_obj_cell_mapping)

    # Create cell summary
    if second_obj_cell_mapping:
        # Group by cell_id once for efficiency
        grouped = second_obj_cell_df.groupby("cell_id")
        cell_summary = []

        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)

            # Initialize basic cell summary
            summary_entry = {
                "cell_id": cell_id,
                "cell_area": cell_area,
            }

            # Check if cell_id has associated secondary objects
            if cell_id in grouped.groups:
                cell_second_objs = grouped.get_group(cell_id)

                # Calculate cell-level statistics
                total_second_obj_area = cell_second_objs["second_obj_area"].sum()
                mean_diameter = cell_second_objs["second_obj_diameter"].mean()

                summary_entry.update(
                    {
                        "has_second_obj": True,
                        "num_second_objs": len(cell_second_objs),
                        "second_obj_ids": list(cell_second_objs["second_obj_id"]),
                        "total_second_obj_area": total_second_obj_area,
                        "second_obj_area_ratio": total_second_obj_area / cell_area
                        if cell_area > 0
                        else 0,
                        "mean_second_obj_diameter": mean_diameter,
                    }
                )

                # Add cell nucleus distance fields if nuclei_centroids was provided
                if nuclei_centroids_dict is not None:
                    mean_distance = (
                        cell_second_objs["distance_to_nucleus"].dropna().mean()
                        if not cell_second_objs["distance_to_nucleus"].dropna().empty
                        else None
                    )
                    summary_entry["mean_distance_to_nucleus"] = mean_distance

            else:  # Cell without secondary objects
                summary_entry.update(
                    {
                        "has_second_obj": False,
                        "num_second_objs": 0,
                        "second_obj_ids": [],
                        "total_second_obj_area": 0,
                        "second_obj_area_ratio": 0,
                        "mean_second_obj_diameter": None,
                    }
                )

                # Add cell nucleus distance fields if nuclei_centroids was provided
                if nuclei_centroids_dict is not None:
                    summary_entry["mean_distance_to_nucleus"] = None

            cell_summary.append(summary_entry)

    else:
        # Handle case with no secondary objects
        cell_summary = []
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            summary_entry = {
                "cell_id": cell_id,
                "has_second_obj": False,
                "num_second_objs": 0,
                "second_obj_ids": [],
                "cell_area": cell_area,
                "total_second_obj_area": 0,
                "second_obj_area_ratio": 0,
                "mean_second_obj_diameter": None,
            }

            # Add cell nucleus distance fields if nuclei_centroids was provided
            if nuclei_centroids_dict is not None:
                summary_entry["mean_distance_to_nucleus"] = None

            cell_summary.append(summary_entry)

    # Create final results
    cell_summary_df = pd.DataFrame(cell_summary)
    cell_second_obj_table = {
        "cell_summary": cell_summary_df,
        "second_obj_cell_mapping": second_obj_cell_df,
    }

    # Create associated secondary object masks
    associated_second_objs = np.zeros_like(labeled_second_objs)
    for mapping in second_obj_cell_mapping:
        second_obj_id = mapping["second_obj_id"]
        second_obj_mask = labeled_second_objs == second_obj_id
        associated_second_objs[second_obj_mask] = second_obj_id

    # Print statistics
    total_kept = len(second_obj_cell_mapping)
    print(
        f"Kept {total_kept} out of {num_second_objs} detected secondary objects "
        f"({total_kept / num_second_objs * 100:.1f}%)"
    )
    print(
        f"Discarded {num_second_objs - total_kept} secondary objects that didn't meet diameter criteria or cell overlap"
    )

    # Process cytoplasm masks if provided
    updated_cytoplasm_masks = None
    if cytoplasm_masks is not None:
        updated_cytoplasm_masks = cytoplasm_masks.copy()
        for mapping in second_obj_cell_mapping:
            second_obj_id = mapping["second_obj_id"]
            cell_id = mapping["cell_id"]
            second_obj_mask = associated_second_objs == second_obj_id
            cytoplasm_mask = updated_cytoplasm_masks == cell_id
            updated_cytoplasm_masks[cytoplasm_mask & second_obj_mask] = 0
        print(
            f"Updated cytoplasm masks by removing {len(second_obj_cell_mapping)} secondary object regions"
        )

    # Return results
    if updated_cytoplasm_masks is not None:
        return (
            associated_second_objs,
            cell_second_obj_table,
            updated_cytoplasm_masks,
        )
    else:
        return associated_second_objs, cell_second_obj_table
