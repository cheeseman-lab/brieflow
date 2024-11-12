"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

import logging
import glob
import re

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
    # Well info
    well = data_location.get("well")
    well_str = f"W{well}" if well else ""

    # Tile info
    tile = data_location.get("tile")
    tile_str = f"_T{tile}" if tile else ""

    # Cycle info
    cycle = data_location.get("cycle")
    cycle_str = f"_C{cycle}" if cycle else ""

    # Construct filename by combining components with info_type and file_type
    if well or tile or cycle:
        filename = f"{well_str}{tile_str}{cycle_str}__{info_type}.{file_type}"
    else:
        filename = f"{info_type}.{file_type}"
    return filename


def parse_filename(filename: str) -> tuple:
    """Parse a structured filename to extract data location, information type, and file type.

    Args:
        filename (str): Structured filename, e.g., 'WA1_T02_C03__cell_features.tsv'.

    Returns:
        tuple: A tuple containing:
            - data_location (dict): Dictionary with keys 'well', 'tile', 'cycle' as applicable.
            - info_type (str): The type of information (e.g., 'cell_features').
            - file_type (str): The file extension/type (e.g., 'tsv').
    """
    # Split the filename into main parts
    base, file_type = filename.rsplit(".", 1)
    parts = base.split("__")

    # Initialize data_location dictionary and variables
    data_location = {}
    info_type = None

    # Parse data location part (e.g., 'WA1_T02_C03')
    if len(parts) == 2:
        location_part, info_type = parts
        elements = location_part.split("_")

        for element in elements:
            if element.startswith("W"):
                data_location["well"] = element[1:]  # remove 'W'
            elif element.startswith("T"):
                data_location["tile"] = int(
                    element[1:]
                )  # remove 'T' and convert to int
            elif element.startswith("C"):
                data_location["cycle"] = int(
                    element[1:]
                )  # remove 'C' and convert to int
    else:
        # If no location part, the first part is the info_type
        info_type = parts[0]

    return data_location, info_type, file_type


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
