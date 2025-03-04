"""This module provides functions for creating montages of cells from imaging data.

Functions include:
- Adding image file paths to a DataFrame for downstream analysis.
- Generating montages of mitotic cells with customizable parameters such as size, shape, and channel selection.
- Adding rectangular bounds to a DataFrame for defining cell regions.
- Creating grid views of sub-images based on TIFF image bounding boxes.
- Tiling arrays to produce montages with flexible grid configurations.

Functions:
    - add_filenames: Add image file paths to a DataFrame based on well and tile information.
    - create_mitotic_cell_montage: Create a montage of cells from a DataFrame with flexible selection and layout options.
    - add_rect_bounds: Add rectangular bounds to a DataFrame for defining regions of interest.
    - grid_view: Generate grid views of sub-images from TIFF images using bounding boxes.
    - create_montage: Tile ND arrays into a montage with specified dimensions.
"""

from itertools import product

import numpy as np
from tifffile import imread

from lib.shared.file_utils import get_filename
from lib.external.cp_emulator import subimage


def add_filenames(merge_data, root_fp, montage_subset=False):
    """Adds an image file path column to the given DataFrame.

    This function generates file paths based on the 'well' and 'tile' columns
    in the DataFrame and adds them as a new column named 'image_path'.

    Args:
        merge_data (pd.DataFrame): DataFrame containing 'well' and 'tile' columns.
        root_fp (Path): Root file path to construct the image file paths.
        montage_subset (bool): For montages only return a subset of the DataFrame.

    Returns:
        pd.DataFrame: The updated DataFrame with an added 'image_path' column.
    """
    merge_data = merge_data.copy()

    merge_data["image_path"] = merge_data.apply(
        lambda row: str(
            root_fp
            / "preprocess"
            / "images"
            / "phenotype"
            / get_filename(
                {"plate": row["plate"], "well": row["well"], "tile": row["tile"]},
                "image",
                "tiff",
            )
        ),
        axis=1,
    )

    # Subset to only data that is required for montage generation
    if montage_subset:
        essential_columns = [
            "gene_symbol_0",
            "sgRNA_0",
            "plate",
            "well",
            "tile",
            "i_0",
            "j_0",
            "image_path",
        ]

        # Only keep columns we need
        merge_data = merge_data[essential_columns]

    return merge_data


def create_cell_montage(
    cell_data,
    channels,
    num_cells=30,
    cell_size=40,
    shape=(3, 10),
    selection_params={"method": "head"},
    coordinate_cols=None,
):
    """Create a montage of cells from DataFrame with flexible parameters.

    Args:
        cell_data (pd.DataFrame): DataFrame with cell data.
        channels (list): List with channel names.
        num_cells (int): Number of cells to include.
        cell_size (int): Size of cell bounds box.
        shape (tuple): Shape of montage grid (rows, cols).
        selection_params (dict): Parameters for cell selection.
            {
                'method': 'random' | 'sorted' | 'head',
                'sort_by': column name if method='sorted',
                'ascending': True/False if method='sorted'
            }
        coordinate_cols (list): Names of coordinate columns for bounds, defaults to ['i_0', 'j_0'].

    Returns:
        dict: Dictionary mapping channels to their montage arrays.
    """
    if coordinate_cols is None:
        coordinate_cols = ["i_0", "j_0"]

    if selection_params is None:
        selection_params = {"method": "head"}

    # Select cells based on parameters
    cell_data_subset = cell_data.copy()
    if selection_params["method"] == "head":
        cell_data_subset = cell_data_subset.head(num_cells)
    elif selection_params["method"] == "random":
        cell_data_subset = cell_data_subset.sample(n=num_cells, random_state=0)
    elif selection_params["method"] == "sorted":
        cell_data_subset = cell_data_subset.sort_values(
            selection_params["sort_by"],
            ascending=selection_params.get("ascending", True),
        ).head(num_cells)
    else:
        raise ValueError("Invalid selection method.")

    # Add bounds
    cell_data_subset = cell_data_subset.pipe(
        add_rect_bounds,
        width=cell_size,
        ij=coordinate_cols,
        bounds_col="bounds",
    )

    # Create grid view for all channels
    cell_grid = grid_view(
        filenames=cell_data_subset["image_path"].tolist(),
        bounds=cell_data_subset["bounds"].tolist(),
        padding=0,
    )

    # compile montages for each channel
    montages = {}

    for index, channel in enumerate(channels):
        channel_montage = create_montage(cell_grid[:, index, :, :], shape=shape)
        montages[channel] = channel_montage

    return montages


def add_rect_bounds(cell_data, width=10, ij="ij", bounds_col="bounds"):
    """Add rectangular bounds to a DataFrame.

    Args:
        cell_data (pandas.DataFrame): DataFrame containing the data.
        width (int, optional): Width of the rectangular bounds. Defaults to 10.
        ij (str or tuple, optional): Column name or tuple of column names representing the
            coordinates in the DataFrame. Defaults to 'ij'.
        bounds_col (str, optional): Name of the column to store the bounds. Defaults to 'bounds'.

    Returns:
        pandas.DataFrame: DataFrame with the rectangular bounds added.

    Notes:
        - This function computes rectangular bounds around coordinates in the DataFrame.
        - It iterates over the 'ij' column (or columns) to extract the coordinates.
        - For each coordinate, it computes the bounds as (i - width, j - width, i + width, j + width).
        - The bounds are stored in a list 'arr'.
        - The DataFrame is then assigned a new column named 'bounds_col' with the computed bounds.
        - The modified DataFrame is returned.
    """
    arr = []

    # Iterate over the DataFrame to compute rectangular bounds
    for i, j in cell_data[list(ij)].values.astype(int):
        arr.append((i - width, j - width, i + width, j + width))

    # Assign the computed bounds to a new column in the DataFrame
    return cell_data.assign(**{bounds_col: arr})


def grid_view(filenames, bounds, padding=40):
    """Generates a grid view of sub-images from a list of TIFF images based on given bounding boxes.

    Args:
        filenames (list): List of paths to TIFF image files.
        bounds (list): List of bounding boxes [(i_min, j_min, i_max, j_max), ...] for each file.
        padding (int): Padding to add around each image. Default is 40.

    Returns:
        np.ndarray: Stacked sub-image array.
    """
    padding = int(padding)
    image_cells = []

    for filename, bound_set in zip(filenames, bounds):
        image = imread(filename)
        # Apply padding to the bounding box

        # Extract sub-image with padding
        # image_cell = image[0, i_min:i_max, j_min:j_max].copy()
        image_cell = subimage(image, bound_set, pad=padding)
        image_cells.append(image_cell)

    return np.stack(image_cells)


def create_montage(arr, shape=None):
    """Tile ND arrays in last two dimensions to create a montage.

    Args:
        arr (list of np.ndarray): List of arrays to be tiled.
        shape (tuple, optional): Desired shape of the montage (rows, columns).

    Returns:
        np.ndarray: Tiled montage of input arrays.

    Notes:
        - First N-2 dimensions must match across all input arrays.
        - Tiles are expanded to max height and width and padded with zeros.
        - If shape is not provided, defaults to square, clipping last row if empty.
        - If shape contains -1, that dimension is inferred.
        - If rows or columns is 1, does not pad zeros in width or height respectively.
    """
    sz = list(zip(*[img.shape for img in arr]))
    h, w, n = max(sz[-2]), max(sz[-1]), len(arr)

    # Determine the shape of the montage
    if not shape:
        nr = nc = int(np.ceil(np.sqrt(n)))
        if (nr - 1) * nc >= n:
            nr -= 1
    elif -1 in shape:
        assert shape[0] != shape[1], (
            "cannot infer both rows and columns, use shape=None for square montage"
        )
        shape = np.array(shape)
        infer, given = int(np.argwhere(shape == -1)), int(np.argwhere(shape != -1))
        shape[infer] = int(np.ceil(n / shape[given]))
        if (shape[infer] - 1) * shape[given] >= n:
            shape[infer] -= 1
        nr, nc = shape
    else:
        nr, nc = shape

    # Handle special case where one dimension is 1
    if 1 in (nr, nc):
        assert nr != nc, "no need to montage a single image"
        shape = np.array((nr, nc))
        single_axis, other_axis = (
            int(np.argwhere(shape == 1)),
            int(np.argwhere(shape != 1)),
        )
        arr_padded = []
        for img in arr:
            sub_size = (h, img.shape[-2])[single_axis], (w, img.shape[-1])[other_axis]
            sub = np.zeros(
                img.shape[:-2] + (sub_size[0],) + (sub_size[1],), dtype=arr[0].dtype
            )
            s = [[None] for _ in img.shape]
            s[-2] = (0, img.shape[-2])
            s[-1] = (0, img.shape[-1])
            sub[tuple(slice(*x) for x in s)] = img
            arr_padded.append(sub)
        M = np.concatenate(arr_padded, axis=(-2 + other_axis))
    else:
        M = np.zeros(arr[0].shape[:-2] + (nr * h, nc * w), dtype=arr[0].dtype)
        for (r, c), img in zip(product(range(nr), range(nc)), arr):
            s = [[None] for _ in img.shape]
            s[-2] = (r * h, r * h + img.shape[-2])
            s[-1] = (c * w, c * w + img.shape[-1])
            M[tuple(slice(*x) for x in s)] = img

    return M
