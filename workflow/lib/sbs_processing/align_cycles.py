import numpy as np


def align_cycles(
    data,
    method="DAPI",
    upsample_factor=2,
    window=2,
    cutoff=1,
    q_norm=70,
    align_within_cycle=True,
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

    method : {'DAPI','SBS_mean'}
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
    if align_within_cycle:
        align_it = lambda x: Align.align_within_cycle(
            x, window=window, upsample_factor=upsample_factor
        )
        aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])

    if method == "DAPI":
        # Align cycles using the DAPI channel
        aligned = Align.align_between_cycles(
            aligned, channel_index=0, window=window, upsample_factor=upsample_factor
        )
    elif method == "SBS_mean":
        # Calculate cycle offsets using the average of SBS channels
        sbs_channels = list(range(n, aligned.shape[1]))
        if remove_for_cycle_alignment is not None:
            sbs_channels.remove(remove_for_cycle_alignment)
        target = Align.apply_window(aligned[:, sbs_channels], window=window).max(axis=1)
        normed = Align.normalize_by_percentile(target, q_norm=q_norm)
        normed[normed > cutoff] = cutoff
        offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
        # Apply cycle offsets to each channel
        for channel in range(aligned.shape[1]):
            if channel >= sum(extras):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)
            else:
                # Don't apply offsets to extra channel in the cycle it was acquired
                extra_idx = list(np.cumsum(extras) > channel).index(True)
                extra_offsets = np.array([offsets[extra_idx]] * aligned.shape[0])
                aligned[:, channel] = Align.apply_offsets(
                    aligned[:, channel], extra_offsets
                )
    else:
        raise ValueError(f'method "{method}" not implemented')

    return aligned
