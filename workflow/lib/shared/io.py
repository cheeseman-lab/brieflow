import numpy as np

def save_stack(name, data, luts=None, display_ranges=None, resolution=1., compress=0, dimensions=None, display_mode='composite', photometric='minisblack'):
    """Saves image data to a TIFF file with optional LUTs and display ranges.

    Args:
        name (Path): Path to the output TIFF file.
        data (np.ndarray): Image data to be saved.
        luts (list): List of LUTs for each channel. Default is None.
        display_ranges (list): List of (min, max) pairs for each channel. Default is None.
        resolution (float): Resolution in microns per pixel. Default is 1.
        compress (int): Compression level. Default is 0 (no compression).
        dimensions (str): String specifying dimensions (e.g., 'TZC'). Default is None.
        display_mode (str): Display mode for the image. Default is 'composite'.
        photometric (str): Photometric interpretation ('minisblack' or 'rgb'). Default is 'minisblack'.

    Returns:
        None
    """
    if not (2 <= data.ndim <= 5):
        raise ValueError(f'Input has shape {data.shape}, but number of dimensions must be in range [2, 5]')

    if data.dtype == np.int64:
        if (data >= 0).all() and (data < 2**16).all():
            data = data.astype(np.uint16)
        else:
            data = data.astype(np.float32)
            print('Cast int64 to float32')
    if data.dtype == np.float64:
        data = data.astype(np.float32)
        # print('Cast float64 to float32')

    if data.dtype == np.bool_:
        data = 255 * data.astype(np.uint8)

    if data.dtype == np.int32:
        if data.min() >= 0 and data.max() < 2**16:
            data = data.astype(np.uint16)
        else:
            raise ValueError('Error casting from np.int32 to np.uint16, data out of range')

    if data.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f'Cannot save data of type {data.dtype}')

    resolution = (1. / resolution,) * 2

    if data.ndim == 2:
        min, max = single_contrast(data, display_ranges)
        description = imagej_description_2D(min, max)
        imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress)
    else:
        nchannels = data.shape[-3]
        luts, display_ranges = infer_luts_display_ranges(data, luts, display_ranges)

        leading_shape = data.shape[:-2]
        if dimensions is None:
            dimensions = 'TZC'[::-1][:len(leading_shape)][::-1]

        if ('C' not in dimensions) or (nchannels == 1):
            contrast = single_contrast(data, display_ranges)
            description = imagej_description(leading_shape, dimensions, contrast=contrast)
            imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress)
        else:
            description = imagej_description(leading_shape, dimensions, display_mode=display_mode)
            tag_50838 = ij_tag_50838(nchannels)
            tag_50839 = ij_tag_50839(luts, display_ranges)

            imsave(name, data, photometric=photometric, description=description, resolution=resolution, compress=compress,
                   extratags=[(50838, 'I', len(tag_50838), tag_50838, True),
                              (50839, 'B', len(tag_50839), tag_50839, True)])
