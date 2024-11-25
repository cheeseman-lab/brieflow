"""Helper function to extract phenotype features from CellProfiler-like data with multi-channel functionality."""

from itertools import combinations, permutations, product

import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology
import skimage.filters
import skimage.feature
import skimage.segmentation
from scipy import ndimage as ndi

import lib.shared.features
from external.cp_emulator import (
    grayscale_features_multichannel,
    correlation_features_multichannel,
    shape_features,
    grayscale_columns_multichannel,
    correlation_columns_multichannel,
    shape_columns,
    neighbor_measurements,
)
from lib.shared.extract_phenotype_minimal import extract_features
from lib.sbs_process.log_filter import log_ndi


def extract_phenotype_cp_multichannel(
    data_phenotype,
    nuclei,
    cells,
    wildcards,
    cytoplasms=None,
    nucleus_channels="all",
    cell_channels="all",
    cytoplasm_channels="all",
    foci_channel=None,
    channel_names=["dapi", "tubulin", "gh2ax", "phalloidin"],
):
    """Extract phenotype features from CellProfiler-like data with multi-channel functionality.

    Parameters:
    - data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
    - nuclei (numpy.ndarray): Nuclei segmentation data.
    - cells (numpy.ndarray): Cell segmentation data.
    - cytoplasms (numpy.ndarray, optional): Cytoplasmic segmentation data.
    - wildcards (dict): Dictionary containing wildcards.
    - nucleus_channels (str or list): List of nucleus channel indices to consider or 'all'.
    - cell_channels (str or list): List of cell channel indices to consider or 'all'.
    - foci_channel (int): Index of the channel containing foci information.
    - channel_names (list): List of channel names.

    Returns:
    - pandas.DataFrame: DataFrame containing extracted phenotype features.
    """
    # Check if all channels should be used
    if nucleus_channels == "all":
        try:
            nucleus_channels = list(range(data_phenotype.shape[-3]))
        except:
            nucleus_channels = [0]

    if cell_channels == "all":
        try:
            cell_channels = list(range(data_phenotype.shape[-3]))
        except:
            cell_channels = [0]

    if cytoplasm_channels == "all":
        try:
            cytoplasm_channels = list(range(data_phenotype.shape[-3]))
        except:
            cytoplasm_channels = [0]

    dfs = []

    # Define features
    features = grayscale_features_multichannel
    features.update(correlation_features_multichannel)
    features.update(shape_features)

    # Define function to create column map
    def make_column_map(channels):
        columns = {}
        # Create columns for grayscale features
        for feat, out in grayscale_columns_multichannel.items():
            columns.update(
                {
                    f"{feat}_{n}": f"{channel_names[ch]}_{renamed}"
                    for n, (renamed, ch) in enumerate(product(out, channels))
                }
            )
        # Create columns for correlation features
        for feat, out in correlation_columns_multichannel.items():
            if feat == "lstsq_slope":
                iterator = permutations
            else:
                iterator = combinations
            columns.update(
                {
                    f"{feat}_{n}": renamed.format(
                        first=channel_names[first], second=channel_names[second]
                    )
                    for n, (renamed, (first, second)) in enumerate(
                        product(out, iterator(channels, 2))
                    )
                }
            )
        # Add shape columns
        columns.update(shape_columns)
        return columns

    # Create column maps for nucleus and cell
    nucleus_columns = make_column_map(nucleus_channels)
    cell_columns = make_column_map(cell_channels)

    # Extract nucleus features
    dfs.append(
        extract_features(
            data_phenotype[..., nucleus_channels, :, :],
            nuclei,
            wildcards,
            features,
            multichannel=True,
        )
        .rename(columns=nucleus_columns)
        .set_index("label")
        .rename(columns=lambda x: "nucleus_" + x if x not in wildcards.keys() else x)
    )

    # Extract cell features
    dfs.append(
        extract_features(
            data_phenotype[..., cell_channels, :, :],
            cells,
            dict(),
            features,
            multichannel=True,
        )
        .rename(columns=cell_columns)
        .set_index("label")
        .add_prefix("cell_")
    )

    # Extract cytoplasmic features if cytoplasms are provided
    if cytoplasms is not None:
        cytoplasmic_columns = make_column_map(cytoplasm_channels)
        dfs.append(
            extract_features(
                data_phenotype[..., cytoplasm_channels, :, :],
                cytoplasms,
                dict(),
                features,
                multichannel=True,
            )
            .rename(columns=cytoplasmic_columns)
            .set_index("label")
            .add_prefix("cytoplasm_")
        )

    # Extract foci features if foci channel is provided
    if foci_channel is not None:
        foci = find_foci(
            data_phenotype[..., foci_channel, :, :], remove_border_foci=True
        )
        dfs.append(
            extract_features_bare(foci, cells, features=lib.shared.features.foci)
            .set_index("label")
            .add_prefix(f"cell_{channel_names[foci_channel]}_")
        )

    # Extract nucleus and cell neighbors
    dfs.append(
        neighbor_measurements(nuclei, distances=[1])
        .set_index("label")
        .add_prefix("nucleus_")
    )

    dfs.append(
        neighbor_measurements(cells, distances=[1])
        .set_index("label")
        .add_prefix("cell_")
    )
    if cytoplasms is not None:
        dfs.append(
            neighbor_measurements(cytoplasms, distances=[1])
            .set_index("label")
            .add_prefix("cytoplasm_")
        )

    # Concatenate data frames and reset index
    return pd.concat(dfs, axis=1, join="outer", sort=True).reset_index()


def extract_features_bare(
    data, labels, features=None, wildcards=None, multichannel=False
):
    """Extract features in dictionary and combine with generic region features.

    Args:
        data (numpy.ndarray): Image data of dimensions (CHANNEL, I, J).
        labels (numpy.ndarray): Labeled segmentation mask defining objects to extract features from.
        features (dict or None): Features to extract and their defining functions. Default is None.
        wildcards (dict or None): Metadata to include in the output table, e.g., well, tile, etc. Default is None.
        multichannel (bool): Flag indicating whether the data has multiple channels.

    Returns:
        pandas.DataFrame: Table of labeled regions in labels with corresponding feature measurements.
    """
    features = features.copy() if features else dict()
    features.update({"label": lambda r: r.label})

    # Choose appropriate feature table based on multichannel flag
    if multichannel:
        from lib.shared.feature_table_utils import (
            feature_table_multichannel as feature_table,
        )
    else:
        from lib.shared.feature_table_utils import feature_table

    # Extract features using the feature table function
    df = feature_table(data, labels, features)

    # Add wildcard metadata to the DataFrame if provided
    if wildcards is not None:
        for k, v in sorted(wildcards.items()):
            df[k] = v

    return df


def find_foci(data, radius=3, threshold=10, remove_border_foci=False):
    """Detect foci in the given image using a white tophat filter and other processing steps.

    Args:
        data (numpy.ndarray): Input image data.
        radius (int, optional): Radius of the disk used in the white tophat filter. Default is 3.
        threshold (float, optional): Threshold value for identifying foci in the processed image. Default is 10.
        remove_border_foci (bool, optional): Flag to remove foci touching the image border. Default is False.

    Returns:
        labeled (numpy.ndarray): Labeled segmentation mask of foci.
    """
    # Apply white tophat filter to highlight foci
    tophat = skimage.morphology.white_tophat(
        data, selem=skimage.morphology.disk(radius)
    )

    # Apply Laplacian of Gaussian to the filtered image
    tophat_log = log_ndi(tophat, sigma=radius)

    # Threshold the image to create a binary mask
    mask = tophat_log > threshold

    # Remove small objects from the mask
    mask = skimage.morphology.remove_small_objects(mask, min_size=(radius**2))

    # Label connected components in the mask
    labeled = skimage.measure.label(mask)

    # Apply watershed algorithm to refine segmentation
    labeled = apply_watershed(labeled, smooth=1)

    if remove_border_foci:
        # Remove foci touching the border
        border_mask = data > 0
        labeled = remove_border(labeled, ~border_mask)

    return labeled


def apply_watershed(img, smooth=4):
    """Apply the watershed algorithm to the given image to refine segmentation.

    Args:
        img (numpy.ndarray): Input binary image.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map. Default is 4.

    Returns:
        result (numpy.ndarray): Labeled image after watershed segmentation.
    """
    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(img)

    if smooth > 0:
        # Apply Gaussian smoothing to the distance transform
        distance = skimage.filters.gaussian(distance, sigma=smooth)

    # Identify local maxima in the distance transform
    local_max = skimage.feature.peak_local_max(
        distance, indices=False, footprint=np.ones((3, 3)), exclude_border=False
    )

    # Label the local maxima
    markers = ndi.label(local_max)[0]

    # Apply watershed algorithm to the distance transform
    result = skimage.segmentation.watershed(-distance, markers, mask=img)

    return result.astype(np.uint16)


def remove_border(labels, mask, dilate=5):
    """Remove labeled regions that touch the border of the given mask.

    Args:
        labels (numpy.ndarray): Labeled image.
        mask (numpy.ndarray): Mask indicating the border regions.
        dilate (int, optional): Number of dilation iterations to apply to the mask. Default is 5.

    Returns:
        labels (numpy.ndarray): Labeled image with border regions removed.
    """
    # Dilate the mask to ensure regions touching the border are included
    mask = skimage.morphology.binary_dilation(mask, np.ones((dilate, dilate)))

    # Identify labels that need to be removed
    remove = np.unique(labels[mask])

    # Remove the identified labels from the labeled image
    labels = labels.copy()
    labels.flat[np.in1d(labels, remove)] = 0

    return labels
