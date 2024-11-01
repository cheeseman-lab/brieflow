import numpy as np
import skimage

from lib.shared.image_utils import applyIJ


def apply_window(data, window):
    """Apply a window to image data.

    Parameters
    ----------
    data : numpy array
        Image data.
    window : int
        Size of the window to apply.

    Returns
    -------
    filtered : numpy array
        Filtered image data.
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

    Parameters
    ----------
    data : numpy array
        Input image data.
    mask : numpy array
        Boolean mask indicating areas to be replaced with noise.
    x1 : int
        Lower threshold value.
    x2 : int
        Upper threshold value.

    Returns
    -------
    filtered : numpy array
        Filtered image data.
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

    Parameters
    ----------
    data_ : numpy array
        Image data.
    upsample_factor : int
        Upsampling factor for cross-correlation.

    Returns
    -------
    offsets : numpy array
        Offset values between images.
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

    Parameters
    ----------
    data : numpy array
        Input image data.
    q1 : int
        Lower percentile threshold.
    q2 : int
        Upper percentile threshold.

    Returns
    -------
    filtered : numpy array
        Filtered image data.
    """
    # Calculate the q1th and q2th percentiles of the input data
    x1, x2 = np.percentile(data, [q1, q2])
    # Create a mask where values are outside the range [x1, x2]
    mask = (x1 > data) | (x2 < data)
    # Fill the masked values with uniform noise in the range [x1, x2] using the fill_noise function
    return fill_noise(data, mask, x1, x2)


def apply_offsets(data_, offsets):
    """Apply offsets to image data.

    Parameters
    ----------
    data_ : numpy array
        Image data.
    offsets : numpy array
        Offset values to be applied.

    Returns
    -------
    warped : numpy array
        Warped image data.
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


def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
    """Align images within the same cycle.

    Parameters
    ----------
    data_ : numpy array
        Image data.
    upsample_factor : int, optional, default: 4
        Upsampling factor for cross-correlation.
    window : int, optional, default: 1
        Size of the window to apply during alignment.
    q1 : int, optional, default: 0
        Lower percentile threshold.
    q2 : int, optional, default: 90
        Upper percentile threshold.

    Returns
    -------
    aligned : numpy array
        Aligned image data.
    """
    # Filter the input data based on percentiles
    filtered = filter_percentiles(apply_window(data_, window), q1=q1, q2=q2)
    # Calculate offsets using the filtered data
    offsets = calculate_offsets(filtered, upsample_factor=upsample_factor)
    # Apply the calculated offsets to the original data and return the result
    return apply_offsets(data_, offsets)


def align_between_cycles(
    data, channel_index, upsample_factor=4, window=1, return_offsets=False
):
    """Align images between different cycles.

    Parameters
    ----------
    data : numpy array
        Image data.
    channel_index : int
        Index of the channel to align between cycles.
    upsample_factor : int, optional, default: 4
        Upsampling factor for cross-correlation.
    window : int, optional, default: 1
        Size of the window to apply during alignment.
    return_offsets : bool, optional, default: False
        Whether to return the calculated offsets.

    Returns
    -------
    aligned : numpy array
        Aligned image data.
    offsets : numpy array, optional
        Calculated offsets if return_offsets is True.
    """
    # Calculate offsets from the target channel
    target = apply_window(data[:, channel_index], window)
    offsets = calculate_offsets(target, upsample_factor=upsample_factor)

    # Apply the calculated offsets to all channels
    warped = []
    for data_ in data.transpose([1, 0, 2, 3]):
        warped += [apply_offsets(data_, offsets)]

    # Transpose the array back to its original shape
    aligned = np.array(warped).transpose([1, 0, 2, 3])

    # Return aligned data with offsets if requested
    if return_offsets:
        return aligned, offsets
    else:
        return aligned


def normalize_by_percentile(data_, q_norm=70):
    """Normalize data by the specified percentile.

    Parameters
    ----------
    data_ : numpy array
        Input image data.
    q_norm : int, optional, default: 70
        Percentile value for normalization.

    Returns
    -------
    normed : numpy array
        Normalized image data.
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


def align_cycles(
    data,
    method="DAPI",
    upsample_factor=2,
    window=2,
    cutoff=1,
    q_norm=70,
    use_align_within_cycle=True,
    cycle_files=None,
    keep_extras=False,
    n=1,
    remove_for_cycle_alignment=None,
):
    """
    Rigid alignment of sequencing cycles and channels.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarrays
        Unaligned SBS image with dimensions (CYCLE, CHANNEL, I, J) or list of single cycle
        SBS images, each with dimensions (CHANNEL, I, J)

    method : {'DAPI','sbs_mean'}
        Method to use for alignment.

    upsample_factor : int, default 2
        Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).

    window : int or float, default 2
        A centered subset of data is used if `window` is greater than one.

    cutoff : int or float, default 1
        Cutoff for normalized data to help deal with noise in images.

    q_norm : int, default 70
        Quantile for normalization to help deal with noise in images.

    align_within_cycle : bool, default True
        Align SBS channels within cycles.

    cycle_files : list of int or None, default None
        Used for parsing sets of images where individual channels are in separate files, which
        is more typically handled in a preprocessing step to combine images from the same cycle.

    keep_extras : bool, default False
        Retain channels that are not common across all cycles by propagating each 'extra' channel
        to all cycles. Ignored if same number of channels exist for all cycles.

    n : int, default 1
        Determines the first SBS channel in `data`. This is after dealing with `keep_extras`, so
        should only account for channels in common across all cycles if `keep_extras`=False.

    remove_for_cycle_alignment : None or int, default int
        Channel index to remove when finding cycle offsets. This is after dealing with `keep_extras`,
        so should only account for channels in common across all cycles if `keep_extras`=False.

    Returns
    -------
    aligned : np.ndarray
        SBS image aligned across cycles.
    """

    # Handle case where cycle_files is provided
    if cycle_files is not None:
        arr = []
        current = 0
        # Iterate through cycle files to de-nest list of numpy arrays
        for cycle in cycle_files:
            if cycle == 1:
                arr.append(data[current])
            else:
                arr.append(np.array(data[current : current + cycle]))
            current += cycle
        data = arr
        print(data[0].shape)
        print(data[1].shape)

    # Check if the number of channels varies across cycles
    if not all(x.shape == data[0].shape for x in data):
        # Keep only channels in common across all cycles
        channels = [x.shape[-3] if x.ndim > 2 else 1 for x in data]
        stacked = np.array([x[-min(channels) :] for x in data])

        # Add back extra channels if requested
        if keep_extras:
            extras = np.array(channels) - min(channels)
            arr = []
            for cycle, extra in enumerate(extras):
                if extra != 0:
                    arr.extend([data[cycle][extra_ch] for extra_ch in range(extra)])
            propagate = np.array(arr)
            stacked = np.concatenate(
                (np.array([propagate] * stacked.shape[0]), stacked), axis=1
            )
        else:
            extras = [0] * stacked.shape[0]
    else:
        stacked = np.array(data)
        extras = [0] * stacked.shape[0]

    assert stacked.ndim == 4, "Input data must have dimensions CYCLE, CHANNEL, I, J"

    # Align between SBS channels for each cycle
    aligned = stacked.copy()
    if use_align_within_cycle:

        def align_it(x):
            return align_within_cycle(x, window=window, upsample_factor=upsample_factor)

        aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])

    if method == "DAPI":
        # Align cycles using the DAPI channel
        aligned = align_between_cycles(
            aligned, channel_index=0, window=window, upsample_factor=upsample_factor
        )
    elif method == "sbs_mean":
        # Calculate cycle offsets using the average of SBS channels
        sbs_channels = list(range(n, aligned.shape[1]))
        if remove_for_cycle_alignment is not None:
            sbs_channels.remove(remove_for_cycle_alignment)
        target = apply_window(aligned[:, sbs_channels], window=window).max(axis=1)
        normed = normalize_by_percentile(target, q_norm=q_norm)
        normed[normed > cutoff] = cutoff
        offsets = calculate_offsets(normed, upsample_factor=upsample_factor)
        # Apply cycle offsets to each channel
        for channel in range(aligned.shape[1]):
            if channel >= sum(extras):
                aligned[:, channel] = apply_offsets(aligned[:, channel], offsets)
            else:
                # Don't apply offsets to extra channel in the cycle it was acquired
                extra_idx = list(np.cumsum(extras) > channel).index(True)
                extra_offsets = np.array([offsets[extra_idx]] * aligned.shape[0])
                aligned[:, channel] = apply_offsets(aligned[:, channel], extra_offsets)
    else:
        raise ValueError(f'method "{method}" not implemented')

    return aligned
