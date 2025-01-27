from itertools import product

import numpy as np
from tifffile import imread


from lib.external.cp_emulator import subimage


def create_mitotic_cell_montage(
    df,
    channels,
    num_cells=30,
    cell_size=40,
    shape=(3, 10),
    selection_params=None,
    coordinate_cols=None,
):
    """
    Create a montage of cells from DataFrame with flexible parameters.
    Designed to save data directly, for use with interactive visualization.

    Args:
        df (pd.DataFrame): DataFrame with cell data
        output_dir (str): Directory to save montages
        output_prefix (str): Prefix for output filenames
        channels (dict): Dictionary mapping channel names to filename column names
                        e.g. {'DAPI': 'filename_DAPI', 'GFP': 'filename_GFP'}
        num_cells (int): Number of cells to include
        cell_size (int): Size of cell bounds box
        shape (tuple): Shape of montage grid (rows, cols)
        selection_params (dict): Parameters for cell selection
                        {
                            'method': 'random' | 'sorted' | 'head',
                            'sort_by': column name if method='sorted',
                            'ascending': True/False if method='sorted'
                        }
        coordinate_cols (list): Names of coordinate columns for bounds, defaults to ['i_0', 'j_0']

    Returns:
        dict: Dictionary mapping channels to their montage arrays
    """
    if coordinate_cols is None:
        coordinate_cols = ["i_0", "j_0"]

    if selection_params is None:
        selection_params = {"method": "head"}

    # Select cells based on parameters
    df_subset = df.copy()
    if selection_params["method"] == "random":
        df_subset = df_subset.sample(n=num_cells)
    elif selection_params["method"] == "sorted":
        df_subset = df_subset.sort_values(
            selection_params["sort_by"],
            ascending=selection_params.get("ascending", True),
        ).head(num_cells)
    else:
        df_subset = df_subset.head(num_cells)

    # Add bounds
    df_subset = df_subset.pipe(
        add_rect_bounds,
        width=cell_size,
        ij=coordinate_cols,
        bounds_col="bounds",
    )

    # TODO: finish testing
    return grid_view(
        filenames=df_subset["image_path"],
        bounds=df_subset["bounds"].tolist(),
        padding=0,
    )

    # Store montages
    montages = {}

    # Create montages for each channel
    for channel_name, channel_info in channels.items():
        # Parse the channel dict
        if isinstance(channel_info, dict):
            filename = channel_info["filename"]
            if filename != "filename":
                filename = f"filename_{filename}"
            channel_idx = channel_info.get("channel")
        else:
            filename = channel_info
            channel_idx = None

        # Create grid
        cell_grid = grid_view(
            files=df_subset[filename].tolist(),
            bounds=df_subset["bounds"].tolist(),
            padding=0,
        )

        # Create montage
        montage = create_montage(cell_grid, shape=shape)
        montages[channel_name] = montage

        # Select channel if specified in channel_info
        if channel_idx is not None:
            montage = montage[channel_idx]

        return montage


def add_rect_bounds(df, width=10, ij="ij", bounds_col="bounds"):
    """
    Add rectangular bounds to a DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame containing the data.
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
    for i, j in df[list(ij)].values.astype(int):
        arr.append((i - width, j - width, i + width, j + width))

    # Assign the computed bounds to a new column in the DataFrame
    return df.assign(**{bounds_col: arr})


def grid_view(filenames, bounds, padding=40, with_mask=False):
    """
    Generates a grid view of sub-images from a list of TIFF images based on given bounding boxes.

    Args:
        filenames (list): List of paths to TIFF image files.
        bounds (list): List of bounding boxes [(i_min, j_min, i_max, j_max), ...] for each file.
        padding (int): Padding to add around each image. Default is 40.
        with_mask (bool): If True, generates a mask image. Default is False.

    Returns:
        np.ndarray: Stacked sub-image array.
        np.ndarray (optional): Stacked mask array if with_mask is True.
    """
    padding = int(padding)
    sub_images = []

    for filename, bound_set in zip(filenames, bounds):
        image = imread(filename)
        for i_min, j_min, i_max, j_max in bound_set:
            # Apply padding to the bounding box
            i_min_p, j_min_p = max(0, i_min - padding), max(0, j_min - padding)
            i_max_p, j_max_p = (
                min(image.shape[0], i_max + padding),
                min(image.shape[1], j_max + padding),
            )

            # Extract sub-image with padding
            sub_image = image[i_min_p:i_max_p, j_min_p:j_max_p].copy()
            sub_images.append(sub_image)

    if with_mask:
        masks = [
            np.full((i_max_p - i_min_p, j_max_p - j_min_p), idx + 1, dtype=np.uint16)
            for idx, bound_set in enumerate(bounds)
            for i_min_p, j_min_p, i_max_p, j_max_p in bound_set
        ]
        return np.stack(sub_images), np.stack(masks)

    return np.stack(sub_images)


def create_montage(arr, shape=None):
    """
    Tile ND arrays in last two dimensions to create a montage.

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
        assert (
            shape[0] != shape[1]
        ), "cannot infer both rows and columns, use shape=None for square montage"
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
