"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

import logging
import glob

import numpy as np

from lib.shared.external.tifffile import TiffFile, TiffSequence

log = logging.getLogger(__name__)


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a structured filename based on data location, information type, and file type.

    Args:
        data_location (dict): Dictionary containing location info like well, tile, and cycle.
        info_type (str): Type of information (e.g., 'cell_features', 'sbs_reads').
        file_type (str): File extension/type (e.g., 'tsv', 'hdf5', 'tiff').

    Returns:
        str: Structured filename.
    """
    # Well has no leading zeros
    well_str = f"W{data_location.get('well')}"

    # Tile with 4 digits leading zero padding if numeric
    tile = data_location.get("tile")
    tile_str = f"_T{tile}" if tile else ""

    # Cycle with 2 digits leading zero padding if numeric
    cycle = data_location.get("cycle")
    cycle_str = f"_C{cycle}" if cycle else ""

    # Construct filename by combining components with info_type and file_type
    filename = f"{well_str}{tile_str}{cycle_str}__{info_type}.{file_type}"
    return filename


def read_stack(filename, copy=True, maxworkers=None, fix_axes=False):
    """Reads a TIFF file into a numpy array, with optional memory mapping and axis fixing.

    Args:
        filename (str): Path to the TIFF file.
        copy (bool): If True, returns a copy of the data. Default is True.
        maxworkers (int): Number of threads for decompression. Default is None.
        fix_axes (bool): If True, fixes incorrect axis orders. Default is False.

    Returns:
        np.ndarray: Image data.
    """
    data = imread_tiff(filename, multifile=False, is_ome=False, maxworkers=maxworkers)

    while data.shape[0] == 1:
        data = np.squeeze(data, axis=(0,))

    if copy:
        data = data.copy()

    if fix_axes:
        if data.ndim != 4:
            raise ValueError("`fix_axes` only tested for data with 4 dimensions")
        data = np.array(
            [
                data.reshape((-1,) + data.shape[-2:])[n :: data.shape[-4]]
                for n in range(data.shape[-4])
            ]
        )

    return data


# NOTE: memoize decorator not migrated from OpticalPooledScreens
# consider migrating if needed
def imread_tiff(files, **kwargs):
    """Return image data from TIFF file(s) as numpy array.

    Refer to the TiffFile and  TiffSequence classes and their asarray
    functions for documentation.

    Parameters
    ----------
    files : str, binary stream, or sequence
        File name, seekable binary stream, glob pattern, or sequence of
        file names.
    kwargs : dict
        Parameters 'name', 'offset', 'size', 'multifile', and 'is_ome'
        are passed to the TiffFile constructor.
        The 'pattern' parameter is passed to the TiffSequence constructor.
        Other parameters are passed to the asarray functions.
        The first image series in the file is returned if no arguments are
        provided.

    """
    kwargs_file = parse_kwargs(
        kwargs,
        "is_ome",
        "multifile",
        "_useframes",
        "name",
        "offset",
        "size",
        "multifile_close",
        "fastij",
        "movie",
    )  # legacy
    kwargs_seq = parse_kwargs(kwargs, "pattern")

    if kwargs.get("pages", None) is not None:
        if kwargs.get("key", None) is not None:
            raise TypeError("the 'pages' and 'key' arguments cannot be used together")
        log.warning("imread: the 'pages' argument is deprecated")
        kwargs["key"] = kwargs.pop("pages")

    basestring = str, bytes
    if isinstance(files, basestring) and any(i in files for i in "?*"):
        files = glob.glob(files)
    if not files:
        raise ValueError("no files found")
    if not hasattr(files, "seek") and len(files) == 1:
        files = files[0]

    if isinstance(files, basestring) or hasattr(files, "seek"):
        with TiffFile(files, **kwargs_file) as tif:
            return tif.asarray(**kwargs)
    else:
        with TiffSequence(files, **kwargs_seq) as imseq:
            return imseq.asarray(**kwargs)


def parse_kwargs(kwargs, *keys, **keyvalues):
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    Existing keys are deleted from kwargs.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    >>> kwargs == {'one': 1}
    True
    >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
    True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
        else:
            result[key] = value
    return result
