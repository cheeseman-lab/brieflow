"""Helper function to extract phenotype features from CellProfiler-like data for vacuoles."""

from itertools import combinations, permutations, product

import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology
import skimage.filters
import skimage.feature
import skimage.segmentation
from scipy import ndimage as ndi

from lib.external.cp_emulator import (
    grayscale_features_multichannel,
    correlation_features_multichannel,
    shape_features,
    grayscale_columns_multichannel,
    correlation_columns_multichannel,
    shape_columns,
    neighbor_measurements,
)
from lib.shared.feature_extraction import extract_features, extract_features_bare
from lib.shared.log_filter import log_ndi
from lib.phenotype.constants import DEFAULT_METADATA_COLS


def extract_phenotype_vacuoles(
    data_phenotype,
    vacuoles,
    wildcards,
    vacuole_cell_mapping_df=None,
    vacuole_channels="all",
    foci_channel=None,
    channel_names=["dapi", "tubulin", "gh2ax", "phalloidin"],
):
    """Extract phenotype features for vacuoles with multi-channel functionality.

    Updated version with proper column ordering matching cp_multichannel.

    Args:
        data_phenotype (numpy.ndarray): Phenotype data array of shape (..., CHANNELS, I, J).
        vacuoles (numpy.ndarray): Vacuole segmentation mask with unique integers for each vacuole.
        vacuole_cell_mapping_df (pandas.DataFrame): DataFrame containing the mapping between vacuoles and cells.
        wildcards (dict): Dictionary containing wildcards.
        vacuole_channels (str or list): List of channel indices to consider for vacuole analysis or 'all'.
        foci_channel (int, optional): Index of the channel containing foci information.
        channel_names (list): List of channel names.

    Returns:
        pandas.DataFrame: DataFrame containing extracted phenotype features for each vacuole.
    """
    # If vacuoles are empty, return an empty DataFrame
    if np.sum(vacuoles) == 0:
        print("No vacuoles found for feature extraction.")
        return pd.DataFrame(columns=["vacuole_id", "cell_id"])

    # Check if all channels should be used
    if vacuole_channels == "all":
        try:
            vacuole_channels = list(range(data_phenotype.shape[-3]))
        except:
            vacuole_channels = [0]

    dfs = []

    # Define features
    features = grayscale_features_multichannel.copy()
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

    # Create column map for vacuoles
    vacuole_columns = make_column_map(vacuole_channels)

    # Extract vacuole features for all channels
    dfs.append(
        extract_features(
            data_phenotype[..., vacuole_channels, :, :],
            vacuoles,
            dict(),  # Pass empty dict instead of wildcards here
            features,
            multichannel=True,
        )
        .rename(columns=vacuole_columns)
        .set_index("label")
        .add_prefix("vacuole_")
    )

    # Extract foci features within vacuoles if foci channel is provided
    if foci_channel is not None:
        foci = find_foci_in_vacuoles(
            data_phenotype[..., foci_channel, :, :], vacuoles, remove_border_foci=True
        )

        if foci is not None:
            dfs.append(
                extract_features_bare(foci, vacuoles, features=foci_features)
                .set_index("label")
                .add_prefix(f"vacuole_{channel_names[foci_channel]}_")
            )

    # Extract vacuole neighbor measurements
    dfs.append(
        neighbor_measurements(vacuoles, distances=[1])
        .set_index("label")
        .add_prefix("vacuole_")
    )

    # Concatenate vacuole features
    vacuole_features = pd.concat(dfs, axis=1, join="outer", sort=False).reset_index()

    # Combine with vacuole_cell_mapping_df
    if vacuole_cell_mapping_df is not None:
        vacuole_df = pd.merge(
            vacuole_cell_mapping_df,
            vacuole_features.rename(columns={"label": "vacuole_id"}),
            on="vacuole_id",
            how="left",
            suffixes=("_map", "_feat"),  # left, right
        )

        # If both exist, make a single vacuole_area column (prefer features)
        # If other features are present in both dataframes, modify the next next four lines to reflect the column names and add _map and _feat suffixes
        if {"vacuole_area_map", "vacuole_area_feat"} <= set(vacuole_df.columns):
            vacuole_df["vacuole_area"] = vacuole_df["vacuole_area_feat"].combine_first(
                vacuole_df["vacuole_area_map"]
            )
            vacuole_df = vacuole_df.drop(
                columns=["vacuole_area_map", "vacuole_area_feat"]
            )
    else:
        vacuole_df = vacuole_features.rename(columns={"label": "vacuole_id"})

    # Add wildcards metadata at the END (they'll be reordered later)
    for k, v in sorted(wildcards.items()):
        vacuole_df[k] = v

    # Apply column ordering
    vacuole_df = order_dataframe_columns_vacuoles(vacuole_df)

    return vacuole_df


def order_dataframe_columns_vacuoles(
    df, metadata_cols=None, label_cols=["vacuole_id", "cell_id"]
):
    """Reorder DataFrame columns to put metadata first, then features for vacuoles.

    Args:
        df (pandas.DataFrame): DataFrame to reorder
        metadata_cols (list): List of metadata column names to put first
        label_cols (list): Names of the label columns (vacuole_id, cell_id)

    Returns:
        pandas.DataFrame: DataFrame with reordered columns
    """
    if metadata_cols is None:
        metadata_cols = DEFAULT_METADATA_COLS

    # Start with label columns
    ordered_cols = []
    for col in label_cols:
        if col in df.columns:
            ordered_cols.append(col)

    # Add metadata columns that exist in the DataFrame
    for col in metadata_cols:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)

    # Categorize remaining feature columns
    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    # Group features by type
    vacuole_features = [col for col in remaining_cols if col.startswith("vacuole_")]

    # Add any other columns that don't fit the above patterns
    other_features = [col for col in remaining_cols if not col.startswith("vacuole_")]

    # Combine in desired order
    ordered_cols.extend(other_features)  # Any additional metadata/wildcards
    ordered_cols.extend(vacuole_features)

    return df[ordered_cols]


def find_foci_in_vacuoles(
    data, vacuoles, radius=3, threshold=10, remove_border_foci=False
):
    """Detect foci within vacuoles using a white tophat filter and other processing steps.

    Args:
        data (numpy.ndarray): Input image data.
        vacuoles (numpy.ndarray): Vacuole segmentation mask.
        radius (int, optional): Radius of the disk used in the white tophat filter. Default is 3.
        threshold (float, optional): Threshold value for identifying foci in the processed image. Default is 10.
        remove_border_foci (bool, optional): Flag to remove foci touching the vacuole border. Default is False.

    Returns:
        labeled (numpy.ndarray): Labeled segmentation mask of foci within vacuoles.
    """
    # If no vacuoles, return None
    if np.sum(vacuoles) == 0:
        return None

    # Create a binary mask for all vacuoles
    vacuole_mask = vacuoles > 0

    # Mask the input data to only consider pixels within vacuoles
    masked_data = np.zeros_like(data)
    masked_data[vacuole_mask] = data[vacuole_mask]

    # Apply white tophat filter to highlight foci
    tophat = skimage.morphology.white_tophat(
        masked_data, footprint=skimage.morphology.disk(radius)
    )

    # Apply Laplacian of Gaussian to the filtered image
    tophat_log = log_ndi(tophat, sigma=radius)

    # Threshold the image to create a binary mask
    mask = tophat_log > threshold

    # Remove small objects from the mask
    mask = skimage.morphology.remove_small_objects(mask, min_size=(radius**2))

    # Ensure we only keep foci within vacuoles
    mask = mask & vacuole_mask

    # Label connected components in the mask
    labeled = skimage.measure.label(mask)

    # Apply watershed algorithm to refine segmentation
    labeled = apply_watershed(labeled, smooth=1)

    if remove_border_foci:
        # Create a border mask for vacuoles
        vacuole_border = skimage.segmentation.find_boundaries(vacuole_mask)
        # Remove foci touching the vacuole border
        labeled = remove_border(labeled, vacuole_border)

    return labeled


def apply_watershed(img, smooth=4):
    """Apply the watershed algorithm to the given image to refine segmentation.

    Args:
        img (numpy.ndarray): Input binary image.
        smooth (float, optional): Size of Gaussian kernel used to smooth the distance map. Default is 4.

    Returns:
        result (numpy.ndarray): Labeled image after watershed segmentation.
    """
    # If empty image, return as is
    if np.sum(img) == 0:
        return img

    # Compute the distance transform of the image
    distance = ndi.distance_transform_edt(img)

    if smooth > 0:
        # Apply Gaussian smoothing to the distance transform
        distance = skimage.filters.gaussian(distance, sigma=smooth)

    # Identify local maxima in the distance transform
    local_max_coords = skimage.feature.peak_local_max(
        distance, footprint=np.ones((3, 3)), exclude_border=False
    )

    # Create a boolean mask for peaks
    local_max = np.zeros_like(distance, dtype=bool)
    if len(local_max_coords) > 0:  # Check if any peaks were found
        local_max[tuple(local_max_coords.T)] = (
            True  # Convert coordinates to a boolean mask
        )

        # Label the local maxima
        markers = ndi.label(local_max)[0]

        # Apply watershed algorithm to the distance transform
        result = skimage.segmentation.watershed(-distance, markers, mask=img)
    else:
        # If no peaks found, return the original image
        result = img

    return result.astype(np.uint16)


def remove_border(labels, mask, dilate=2):
    """Remove labeled regions that touch the border of the given mask.

    Args:
        labels (numpy.ndarray): Labeled image.
        mask (numpy.ndarray): Mask indicating the border regions.
        dilate (int, optional): Number of dilation iterations to apply to the mask. Default is 2.

    Returns:
        labels (numpy.ndarray): Labeled image with border regions removed.
    """
    # Dilate the mask to ensure regions touching the border are included
    if dilate > 0:
        mask = skimage.morphology.binary_dilation(mask, np.ones((dilate, dilate)))

    # Identify labels that need to be removed
    remove = np.unique(labels[mask])

    # Remove the identified labels from the labeled image
    labels = labels.copy()
    labels.flat[np.in1d(labels, remove)] = 0

    return labels


# Define foci features specific to vacuoles
foci_features = {
    "foci_count": lambda r: count_labels(r.intensity_image),
    "foci_area": lambda r: (r.intensity_image > 0).sum(),
    "foci_area_ratio": lambda r: (r.intensity_image > 0).sum() / r.area
    if r.area > 0
    else 0,
}


def count_labels(labels, return_list=False):
    """Count the unique non-zero labels in a labeled segmentation mask.

    Args:
        labels (numpy array): Labeled segmentation mask.
        return_list (bool): Flag indicating whether to return the list of unique labels along with the count.

    Returns:
        int or tuple: Number of unique non-zero labels. If return_list is True, returns a tuple containing the count
      and the list of unique labels.
    """
    # Get unique labels in the segmentation mask
    uniques = np.unique(labels)
    # Remove the background label (0)
    ls = np.delete(uniques, np.where(uniques == 0))
    # Count the unique non-zero labels
    num_labels = len(ls)
    # Return the count or both count and list of unique labels based on return_list flag
    if return_list:
        return num_labels, ls
    return num_labels
