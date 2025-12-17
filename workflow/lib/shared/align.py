"""Shared functions for aligning images.

Uses NumPy and scikit-image to provide image
alignment between sequencing cycles, apply percentile-based filtering, fill masked
areas with noise, and perform various transformations to enhance image data quality.
"""

import numpy as np
import skimage

from lib.shared.image_utils import applyIJ, remove_channels


def apply_window(data, window):
    """Apply a window to image data.

    Args:
        data (np.ndarray): Image data.
        window (int): Size of the window to apply.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Extract height and width dimensions from the last two axes of the data shape
    height, width = data.shape[-2:]

    # Define a function to find the border based on the window size
    def find_border(x):
        return int((x / 2.0) * (1 - 1 / float(window)))

    # Calculate border indices
    i, j = find_border(height), find_border(width)

    # Return the data with the border cropped out
    return data[..., i : height - i, j : width - j]


def fill_noise(data, mask, x1, x2):
    """Fill masked areas of data with uniform noise.

    Args:
        data (np.ndarray): Input image data.
        mask (np.ndarray): Boolean mask indicating areas to be replaced with noise.
        x1 (int): Lower threshold value.
        x2 (int): Upper threshold value.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Make a copy of the original data
    filtered = data.copy()
    # Initialize a random number generator with seed 0
    rs = np.random.RandomState(0)
    # Replace the masked values with uniform noise generated in the range [x1, x2]
    filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
    # Return the filtered data
    return filtered


def calculate_offsets(data_, upsample_factor):
    """Calculate offsets between images using phase cross-correlation.

    Args:
        data_ (np.ndarray): Image data.
        upsample_factor (int): Upsampling factor for cross-correlation.

    Returns:
        np.ndarray: Offset values between images.
    """
    # Set the target frame as the first frame in the data
    target = data_[0]
    # Initialize an empty list to store offsets
    offsets = []
    # Iterate through each frame in the data
    for i, src in enumerate(data_):
        # If it's the first frame, add a zero offset
        if i == 0:
            offsets += [(0, 0)]
        else:
            # Calculate the offset between the current frame and the target frame
            offset, _, _ = skimage.registration.phase_cross_correlation(
                src, target, upsample_factor=upsample_factor
            )
            # Add the offset to the list
            offsets += [offset]
    # Convert the list of offsets to a numpy array and return
    return np.array(offsets)


@applyIJ
def filter_percentiles(data, q1, q2):
    """Replace data outside of the percentile range [q1, q2] with uniform noise.

    Args:
        data (np.ndarray): Input image data.
        q1 (int): Lower percentile threshold.
        q2 (int): Upper percentile threshold.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Calculate the q1th and q2th percentiles of the input data
    x1, x2 = np.percentile(data, [q1, q2])
    # Create a mask where values are outside the range [x1, x2]
    mask = (x1 > data) | (x2 < data)
    # Fill the masked values with uniform noise in the range [x1, x2] using the fill_noise function
    return fill_noise(data, mask, x1, x2)


def apply_offsets(data_, offsets):
    """Apply offsets to image data.

    Args:
        data_ (np.ndarray): Image data.
        offsets (np.ndarray): Offset values to be applied.

    Returns:
        np.ndarray: Warped image data.
    """
    # Initialize an empty list to store warped frames
    warped = []
    # Iterate through each frame and its corresponding offset
    for frame, offset in zip(data_, offsets):
        # If the offset is zero, add the frame as it is
        if offset[0] == 0 and offset[1] == 0:
            warped += [frame]
        else:
            # Otherwise, apply a similarity transform to warp the frame based on the offset
            st = skimage.transform.SimilarityTransform(translation=offset[::-1])
            frame_ = skimage.transform.warp(frame, st, preserve_range=True)
            # Add the warped frame to the list
            warped += [frame_.astype(data_.dtype)]
    # Convert the list of warped frames to a numpy array and return
    return np.array(warped)


def normalize_by_percentile(data_, q_norm=70):
    """Normalize data by the specified percentile.

    Args:
        data_ (np.ndarray): Input image data.
        q_norm (int, optional): Percentile value for normalization. Defaults to 70.

    Returns:
        np.ndarray: Normalized image data.
    """
    # Get the shape of the input data
    shape = data_.shape
    # Replace the last two dimensions with a single dimension to allow percentile calculation
    shape = shape[:-2] + (-1,)
    # Calculate the q_normth percentile along the last two dimensions of the data
    p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
    # Normalize the data by dividing it by the calculated percentile values
    normed = data_ / p
    # Return the normalized data
    return normed


def apply_custom_offsets(data, offsets_dict):
    """Apply custom offsets to specific channels in image data.

    Applies custom offsets to specified channels. Useful for aligning channels acquired
    in different imaging rounds when automatic alignment fails or there is no common channel
    to use as reference.

    Offset directions:
    - To shift left: +x
    - To shift right: -x
    - To shift up: +y
    - To shift down: -y

    Args:
        data (np.ndarray): Input image data.
        offsets_dict (dict): Mapping of channel index to (y, x) offset.

    Returns:
        np.ndarray: Image data with custom offsets applied.
    """
    # Set up full offsets array, initialized with zeros
    offsets = np.array([(0, 0) for _ in range(data.shape[0])])

    # Apply custom offsets for specified channels
    for channel, offset_yx in offsets_dict.items():
        offsets[channel] = offset_yx

    # Apply the calculated offsets to data
    adjusted = apply_offsets(data, offsets)

    return adjusted


def calculate_rotation_offset(target, source, upsample_factor=10):
    """Calculate rotation angle between two images using log-polar transform.

    Uses the property that rotation in Cartesian space becomes translation
    in polar space. Phase cross-correlation in polar space finds the rotation.

    Args:
        target (np.ndarray): Reference image (2D).
        source (np.ndarray): Image to align to target (2D).
        upsample_factor (int, optional): Subpixel precision for angle detection.
            Higher values give finer angle resolution. Defaults to 10.

    Returns:
        float: Rotation angle in degrees. Positive = counter-clockwise rotation
            needed to align source to target.
    """
    from skimage.transform import warp_polar

    # Use smaller of image dimensions for radius
    radius = min(target.shape) // 2

    # Convert to polar coordinates (rotation becomes horizontal translation)
    target_polar = warp_polar(target, radius=radius, scaling='linear')
    source_polar = warp_polar(source, radius=radius, scaling='linear')

    # Phase correlation in polar space finds rotation as shift in angle axis
    shift, _, _ = skimage.registration.phase_cross_correlation(
        target_polar, source_polar, upsample_factor=upsample_factor
    )

    # Convert pixel shift to angle (full image height = 360 degrees)
    angle = (shift[0] / target_polar.shape[0]) * 360

    return angle


def calculate_rotation_and_translation_offsets(data_full, data_windowed=None, upsample_factor=4, max_rotation=5.0, log_sigma=3.0):
    """Calculate rotation and translation offsets for a stack of images.

    Uses a multi-step approach for correct coordinate handling:
    1. Compute rough translation on windowed data (center region)
    2. Detect rotation on roughly-aligned, LoG-filtered full image
    3. Apply rotation to ORIGINAL source, then compute final translation

    This ensures translation is computed in the rotated coordinate frame,
    matching the order of operations in apply_rotation_and_translation().

    Args:
        data_full (np.ndarray): Full images with shape (N, H, W) - used for rotation detection.
        data_windowed (np.ndarray, optional): Windowed (cropped) images for translation detection.
            If None, uses data_full for both. Defaults to None.
        upsample_factor (int, optional): Subpixel precision. Defaults to 4.
        max_rotation (float, optional): Maximum allowed rotation in degrees.
            Rotations exceeding this are clamped to 0 (likely spurious).
            Defaults to 5.0 degrees.
        log_sigma (float, optional): Sigma for Laplacian of Gaussian filter applied before
            rotation detection. Larger values (2-4) enhance larger features for more robust
            rotation signal. Defaults to 3.0.

    Returns:
        tuple: (angles, offsets) where:
            - angles: np.ndarray of rotation angles in degrees (N,)
            - offsets: np.ndarray of (dy, dx) translation offsets (N, 2)
    """
    from skimage.transform import rotate
    from scipy import ndimage

    def apply_log_filter(img, sigma):
        """Apply Laplacian of Gaussian to enhance spots/edges."""
        # LoG filter: negative to make spots bright (same as SBS pipeline)
        filtered = -1 * ndimage.gaussian_laplace(img.astype(float), sigma)
        # Clip negative values
        filtered = np.clip(filtered, 0, None)
        return filtered

    # Use windowed data for translation if provided, otherwise use full
    if data_windowed is None:
        data_windowed = data_full

    target_full = data_full[0]
    target_windowed = data_windowed[0]

    # Apply LoG filter to target for rotation detection
    target_full_log = apply_log_filter(target_full, log_sigma)

    angles = [0.0]
    offsets = [(0, 0)]

    for i, (src_full, src_windowed) in enumerate(zip(data_full[1:], data_windowed[1:]), start=1):
        # Step 1: Rough translation on ORIGINAL windowed images
        rough_offset, _, _ = skimage.registration.phase_cross_correlation(
            src_windowed, target_windowed, upsample_factor=upsample_factor
        )

        # Step 2: Apply rough translation to FULL image for rotation detection
        st = skimage.transform.SimilarityTransform(translation=rough_offset[::-1])
        src_full_aligned = skimage.transform.warp(src_full, st, preserve_range=True)

        # Step 3: Apply LoG filter to enhance spots/edges for rotation detection
        src_full_log = apply_log_filter(src_full_aligned, log_sigma)

        # Step 4: Find rotation on LoG-filtered, roughly-aligned image
        angle = calculate_rotation_offset(target_full_log, src_full_log, upsample_factor=upsample_factor)

        # Negate angle: we found how much src is rotated relative to target,
        # so we need to rotate by -angle to bring src back to match target
        angle = -angle

        # Clamp implausible rotations (likely spurious)
        if abs(angle) > max_rotation:
            print(f"  Warning: Cycle {i} rotation {angle:.2f}° exceeds max_rotation={max_rotation}°, setting to 0°")
            angle = 0.0

        angles.append(angle)

        # Step 5: Apply rotation to ORIGINAL source, then compute FINAL translation
        # This ensures translation is computed in the correct (rotated) coordinate frame
        if angle != 0:
            src_windowed_rotated = rotate(src_windowed, angle, preserve_range=True)
        else:
            src_windowed_rotated = src_windowed

        # Compute final translation on ROTATED source (fresh computation, not additive!)
        final_offset, _, _ = skimage.registration.phase_cross_correlation(
            src_windowed_rotated, target_windowed, upsample_factor=upsample_factor
        )

        offsets.append(tuple(final_offset))

    return np.array(angles), np.array(offsets)


def apply_rotation_and_translation(data_, angles, offsets):
    """Apply rotation and translation transforms to image stack.

    Args:
        data_ (np.ndarray): Image stack with shape (N, H, W) or (N, C, H, W).
        angles (np.ndarray): Rotation angles in degrees for each image.
        offsets (np.ndarray): Translation offsets (dy, dx) for each image.

    Returns:
        np.ndarray: Transformed image stack with same shape as input.
    """
    from skimage.transform import rotate

    warped = []
    for i, frame in enumerate(data_):
        angle = angles[i]
        offset = offsets[i]

        # Skip transform if no rotation or translation needed
        if angle == 0 and offset[0] == 0 and offset[1] == 0:
            warped.append(frame)
            continue

        # Handle multi-channel frames (C, H, W)
        if frame.ndim == 3:
            frame_transformed = []
            for channel in frame:
                # Apply rotation first
                if angle != 0:
                    channel = rotate(channel, angle, preserve_range=True)
                # Apply translation
                if offset[0] != 0 or offset[1] != 0:
                    st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                    channel = skimage.transform.warp(channel, st, preserve_range=True)
                frame_transformed.append(channel.astype(data_.dtype))
            warped.append(np.array(frame_transformed))
        else:
            # Single channel (H, W)
            frame_out = frame.copy()
            if angle != 0:
                frame_out = rotate(frame_out, angle, preserve_range=True)
            if offset[0] != 0 or offset[1] != 0:
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_out = skimage.transform.warp(frame_out, st, preserve_range=True)
            warped.append(frame_out.astype(data_.dtype))

    return np.array(warped)


def calculate_dapi_edge_offsets(data, upsample_factor=4, max_rotation=5.0, edge_sigma=2.0):
    """Calculate rotation and translation offsets using edge detection on DAPI.

    Uses Canny edge detection which is robust to intensity changes and saturation,
    making it suitable for DAPI signals that degrade over cycles.

    Args:
        data (np.ndarray): DAPI images with shape (N, H, W).
        upsample_factor (int, optional): Subpixel precision. Defaults to 4.
        max_rotation (float, optional): Maximum allowed rotation in degrees.
            Defaults to 5.0.
        edge_sigma (float, optional): Gaussian sigma for Canny edge detection.
            Larger values detect coarser edges. Defaults to 2.0.

    Returns:
        tuple: (angles, offsets) where:
            - angles: np.ndarray of rotation angles in degrees (N,)
            - offsets: np.ndarray of (dy, dx) translation offsets (N, 2)
    """
    from skimage.feature import canny
    from skimage.transform import rotate

    def get_edges(img, sigma):
        """Get edges using Canny, robust to intensity variations."""
        # Normalize to 0-1 range using percentiles (robust to outliers/saturation)
        img_norm = img.astype(float)
        vmin, vmax = np.percentile(img_norm, [1, 99])
        if vmax > vmin:
            img_norm = np.clip((img_norm - vmin) / (vmax - vmin), 0, 1)
        else:
            img_norm = img_norm - img_norm.min()

        # Canny edge detection
        edges = canny(img_norm, sigma=sigma).astype(float)
        return edges

    target = data[0]
    target_edges = get_edges(target, edge_sigma)

    angles = [0.0]
    offsets = [(0, 0)]

    for i, src in enumerate(data[1:], start=1):
        src_edges = get_edges(src, edge_sigma)

        # Step 1: Rough translation on edge images
        rough_offset, _, _ = skimage.registration.phase_cross_correlation(
            src_edges, target_edges, upsample_factor=upsample_factor
        )

        # Step 2: Apply rough translation for rotation detection
        st = skimage.transform.SimilarityTransform(translation=rough_offset[::-1])
        src_edges_aligned = skimage.transform.warp(src_edges, st, preserve_range=True)

        # Step 3: Detect rotation using log-polar on edge images
        angle = calculate_rotation_offset(target_edges, src_edges_aligned, upsample_factor=upsample_factor)
        angle = -angle  # Negate for correct direction

        if abs(angle) > max_rotation:
            print(f"  Warning: Cycle {i} rotation {angle:.2f}° exceeds max, setting to 0°")
            angle = 0.0

        angles.append(angle)

        # Step 4: Apply rotation to original edges, recompute translation
        if angle != 0:
            src_edges_rotated = rotate(src_edges, angle, preserve_range=True)
        else:
            src_edges_rotated = src_edges

        final_offset, _, _ = skimage.registration.phase_cross_correlation(
            src_edges_rotated, target_edges, upsample_factor=upsample_factor
        )

        offsets.append(tuple(final_offset))

    return np.array(angles), np.array(offsets)
