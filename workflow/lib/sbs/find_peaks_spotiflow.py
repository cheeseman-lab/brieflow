"""Find SBS peaks using Spotiflow."""

import numpy as np
from spotiflow.model import Spotiflow

def find_peaks_spotiflow(
    aligned_images,
    model="general",
    normalization="max",
    prob_thresh=0.5,
    min_distance=1,
    subpixel_precision=True,
    return_details=False,
    verbose=True,
    round_coords=True
):
    """Detect peaks in aligned SBS images using Spotiflow.
    
    Args:
    -----------
    aligned_images : numpy.ndarray
        Aligned SBS images with shape (cycles, channels, height, width).
        First channel (index 0) is assumed to be DAPI and is excluded from processing.
    
    model : str or spotiflow.model.Spotiflow
        Either a pre-loaded Spotiflow model or a string specifying which model to load.
        Available pre-trained models:
        - "general": Default model trained on diverse datasets (pixel sizes: 0.04-0.34µm)
        - "hybiss": Trained on HybISS data from 3 microscopes (pixel sizes: 0.15-0.34µm)
        - "synth_complex": Trained on synthetic data with aberrations (pixel size: 0.1µm)
        - "synth_3d": Trained on synthetic 3D data with Z-artifacts (voxel: 0.2µm)
        - "smfish_3d": Fine-tuned from synth_3d on smFISH 3D data (voxel: 0.13µm YX, 0.48µm Z)
    
    normalization : str
        Method for creating the input to Spotiflow:
        - "max": Maximum projection across cycles and channels (default)
        - "std": Standard deviation projection across cycles and channels
    
    prob_thresh : float
        Probability threshold for spot detection (default: 0.5)
    
    min_distance : int
        Minimum distance between spots in pixels (default: 1)
    
    subpixel_precision : bool
        Whether to use subpixel precision for spot detection (default: True)
    
    return_details : bool
        Whether to return additional details from Spotiflow (default: False)
    
    verbose : bool
        Whether to print progress information (default: True)
    
    round_coords : bool
        Whether to round coordinates to integers (default: True)
    
    Returns:
    --------
    peaks : numpy.ndarray
        Binary array of same shape as input projection (height, width) where 1 indicates
        a peak location and 0 indicates no peak.
    
    details : dict (optional)
        Additional details from Spotiflow if return_details is True
    """
    # Load model if string is provided
    if isinstance(model, str):
        if verbose:
            print(f"Loading Spotiflow '{model}' model...")
        model = Spotiflow.from_pretrained(model)
        
    if verbose:
        print("Detecting candidate reads...")
    
    # Extract base channels (excluding DAPI if that's in the first channel)
    base_channels = aligned_images[:, 1:, :, :]  # All cycles, base channels only
    
    # Get image dimensions
    height, width = base_channels.shape[-2:]
    
    # Prepare input for Spotiflow based on normalization method
    if normalization.lower() == "std":
        if verbose:
            print("Computing standard deviation over all cycles and channels...")
        # Standard deviation across cycles and channels
        projection = np.std(base_channels, axis=(0, 1))
        
    elif normalization.lower() == "max":
        if verbose:
            print("Max projecting across cycles and channels...")
        # First, take maximum across cycles for each channel
        max_across_cycles = np.max(base_channels, axis=0)
        # Then, take maximum across channels
        projection = np.max(max_across_cycles, axis=0)
        
    else:
        raise ValueError(f"Unknown normalization method: {normalization}. Use 'max' or 'std'.")
    
    # Normalize the projection for Spotiflow input
    input_norm = (projection - np.min(projection)) / (np.max(projection) - np.min(projection) + 1e-6)
    
    # Run Spotiflow spot detection
    if verbose:
        print("Running Spotiflow spot detection...")
    
    peak_coords, details = model.predict(
        input_norm,
        prob_thresh=prob_thresh,
        min_distance=min_distance,
        subpix=subpixel_precision,
        verbose=verbose
    )
    
    # Round coordinates to integers if requested
    if round_coords:
        peak_coords = np.array([(int(y), int(x)) for y, x in zip(peak_coords[:, 0], peak_coords[:, 1])])
    
    # Create binary peak array
    peaks = np.zeros((height, width), dtype=np.int8)
    
    # Set peak locations to 1
    # Only use coordinates that are within bounds
    valid_peaks = (peak_coords[:, 0] >= 0) & (peak_coords[:, 0] < height) & \
                 (peak_coords[:, 1] >= 0) & (peak_coords[:, 1] < width)
    valid_coords = peak_coords[valid_peaks]
    peaks[valid_coords[:, 0], valid_coords[:, 1]] = 1
    
    if verbose:
        print(f"{len(peak_coords)} spots detected")
    
    if return_details:
        return peaks, details
    else:
        return peaks