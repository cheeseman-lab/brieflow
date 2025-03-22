"""Find SBS peaks using Spotiflow."""

import numpy as np
from scipy.spatial.distance import cdist
from spotiflow.model import Spotiflow


def find_peaks_spotiflow(
    aligned_images,
    cycle_idx=0,
    model="general",
    prob_thresh=0.5,
    min_distance=3,
    subpixel_precision=True,
    verbose=True,
    round_coords=True
):
    """Detect peaks separately for each base channel and combine results.
    
    Args:
    -----------
    aligned_images : numpy.ndarray
        Aligned SBS images with shape (cycles, channels, height, width).
        First channel (index 0) is assumed to be DAPI and is excluded from processing.
    
    cycle_idx : int
        Index of the cycle to use for peak detection (default: 0, first cycle)
    
    model : str or spotiflow.model.Spotiflow
        Spotiflow model specification
    
    prob_thresh : float
        Probability threshold for spot detection (default: 0.5)
    
    min_distance : int
        Minimum distance between spots in pixels (default: 3)
    
    subpixel_precision : bool
        Whether to use subpixel precision for spot detection (default: True)
    
    verbose : bool
        Whether to print progress information (default: True)
    
    round_coords : bool
        Whether to round coordinates to integers (default: True)
    
    Returns:
    --------
    peaks : numpy.ndarray
        Binary array of shape (height, width) where 1 indicates
        a peak location and 0 indicates no peak.
    
    all_base_coords : list
        List of peak coordinates for each base channel.
    """   
    # Load model if string is provided
    if isinstance(model, str):
        if verbose:
            print(f"Loading Spotiflow '{model}' model...")
        model = Spotiflow.from_pretrained(model)
    
    if verbose:
        print(f"Detecting peaks for each base channel using cycle {cycle_idx}...")
    
    # Get spots for cycle cycle_idx
    spots = aligned_images[cycle_idx, 1:, :, :]  # Selected cycle, base channels only
    
    # Get dimensions
    n_bases, height, width = spots.shape
    
    # Initialize output array
    peaks = np.zeros((height, width), dtype=np.int8)
    
    # List to store coordinates for each base
    all_base_coords = []
    
    # Process each base channel separately
    for base_idx in range(n_bases):
        if verbose:
            print(f"Processing base channel {base_idx+1}/{n_bases}...")
        
        # Extract data for current base
        base_data = spots[base_idx, :, :]
        
        # Use the base_data directly - no normalization
        # Run Spotiflow spot detection
        peak_coords, _ = model.predict(
            base_data,
            prob_thresh=prob_thresh,
            min_distance=min_distance,
            subpix=subpixel_precision,
            verbose=False
        )
        
        # Round coordinates if requested
        if round_coords:
            peak_coords = np.array([(int(y), int(x)) for y, x in zip(peak_coords[:, 0], peak_coords[:, 1])])
        
        # Filter out coordinates outside image boundaries
        valid_peaks = (peak_coords[:, 0] >= 0) & (peak_coords[:, 0] < height) & \
                     (peak_coords[:, 1] >= 0) & (peak_coords[:, 1] < width)
        valid_coords = peak_coords[valid_peaks]
        
        if verbose:
            print(f"  Base {base_idx+1}: {len(valid_coords)} spots detected")
            
        # Store coordinates for this base
        all_base_coords.append(valid_coords)
    
    # Combine results from all bases while enforcing minimum distance
    if verbose:
        print("Combining results from all bases...")
    
    # First, collect all coordinates
    all_coords = np.vstack(all_base_coords) if all_base_coords and all(len(coords) > 0 for coords in all_base_coords) else np.empty((0, 2))
    
    if len(all_coords) > 0:
        # Remove duplicates (exact same positions)
        all_coords = np.unique(all_coords, axis=0)
        
        # Initialize list for final coordinates
        final_coords = []
        
        # Sort coordinates by intensity if available
        # For simplicity, we'll just process them in order
        remaining_coords = all_coords.copy()
        
        while len(remaining_coords) > 0:
            # Take the first coordinate as a "seed"
            seed_coord = remaining_coords[0]
            final_coords.append(seed_coord)
            
            # Calculate distances to all remaining coordinates
            distances = cdist([seed_coord], remaining_coords)
            
            # Find coordinates that are far enough from the seed
            far_enough = distances[0] > min_distance
            
            # Update remaining coords to only those far enough from the seed
            remaining_coords = remaining_coords[far_enough]
        
        # Convert to numpy array
        final_coords = np.array(final_coords)
        
        # Create binary peak array
        peaks = np.zeros((height, width), dtype=np.int8)
        peaks[final_coords[:, 0], final_coords[:, 1]] = 1
        
        if verbose:
            print(f"Final result: {len(final_coords)} spots after enforcing minimum distance of {min_distance}")
    else:
        if verbose:
            print("No spots detected in any channel")
        final_coords = np.empty((0, 2))
    
    return peaks, all_base_coords