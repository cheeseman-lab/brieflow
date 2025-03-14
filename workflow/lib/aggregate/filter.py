"""This module provides functions for loading and formatting data during aggregation.

Functions include:
- Loading a subset of data from parquet files for efficient processing.

Functions:
    - load_parquet_subset: Load a fixed number of random rows from a parquet file.
"""

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.covariance import EllipticEnvelope


def perturbation_filter(
    cell_data,
    perturbation_name_col,
    perturbation_multi_col=None,
    filter_single_pert=False,
):
    """Clean cell data by removing cells without perturbation assignments and optionally filtering for single-gene cells.

    Args:
        cell_data (pd.DataFrame): Raw dataframe containing cell measurements.
        perturbation_name_col (str): Column name containing perturbation assignments.
        perturbation_multi_col (str): Column name containing multi-perturbation flags. Defaults to None.
        filter_single_pert (bool): If True, only keep cells with single-gene perturbations.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Remove cells without perturbation assignments
    clean_cell_data = cell_data[cell_data[perturbation_name_col].notna()].copy()
    print(f"Found {len(clean_cell_data)} cells with assigned perturbations")

    if filter_single_pert:
        # Filter for single-gene cells if requested
        clean_cell_data = clean_cell_data[
            clean_cell_data[perturbation_multi_col] == True
        ]
        print(f"Kept {len(clean_cell_data)} cells with single gene assignments")
    else:
        # Warn about multi-gene cells if not filtering
        multi_pert_cells = len(
            clean_cell_data[clean_cell_data[perturbation_multi_col] == False]
        )
        if multi_pert_cells > 0:
            print(
                f"WARNING: {multi_pert_cells} cells have multiple perturbation assignments"
            )

    return clean_cell_data.reset_index(drop=True)


def missing_values_filter(
    cell_data,
    first_feature,
    impute=True,
    drop_rows=False,
    drop_cols=False,
    drop_cols_threshold=None,
):
    """Filter cell data by handling missing values through dropping or imputation.

    Args:
        cell_data (pd.DataFrame): Raw dataframe containing cell measurements.
        first_feature (str): Name of the first feature column.
        impute (bool): Whether to impute remaining missing values after dropping. Defaults to True.
        drop_rows (bool): Whether to drop all rows with any missing values. Defaults to False.
        drop_cols (bool): Whether to drop all columns with any missing values. Defaults to False.
        drop_cols_threshold (float, optional): If provided, drops columns with NaN proportion >= threshold.
                                              This overrides drop_cols if both are specified.
                                              Range: 0.0-1.0. Defaults to None.

    Returns:
        pd.DataFrame: Filtered dataframe with handled missing values.
    """
    # Get features
    feature_start_idx = cell_data.columns.get_loc(first_feature)
    metadata = cell_data.iloc[:, :feature_start_idx].copy()
    features = cell_data.iloc[:, feature_start_idx:].copy()

    # Get columns with missing values
    cols_with_na = features.columns[features.isna().any()].tolist()

    if not cols_with_na:
        return cell_data

    # Perform dropping operations if requested
    if drop_rows:
        # Drop rows with any missing values
        original_row_count = features.shape[0]
        features.dropna(axis=0, inplace=True)
        print(
            f"Dropped {original_row_count - features.shape[0]} rows with missing values"
        )

        # Update metadata to match remaining rows
        metadata = metadata.loc[features.index]

    # Handle column dropping based on parameters
    if drop_cols_threshold is not None:
        # Calculate proportion of NaN values in each column
        na_proportions = features.isna().mean()

        # Identify columns to drop based on threshold
        cols_to_drop = na_proportions[
            na_proportions >= drop_cols_threshold
        ].index.tolist()

        if cols_to_drop:
            print(
                f"Dropping {len(cols_to_drop)} columns with ≥{drop_cols_threshold * 100}% missing values"
            )
            features.drop(columns=cols_to_drop, inplace=True)

    if drop_cols:
        # Drop all columns with any missing values
        print(f"Dropping all {len(cols_with_na)} columns with any missing values")
        features.drop(columns=cols_with_na, inplace=True)

    # Impute remaining missing values if requested
    if impute:
        # Get updated list of columns with missing values
        remaining_cols_with_na = features.columns[features.isna().any()].tolist()

        if remaining_cols_with_na:
            print(
                f"Imputing {len(remaining_cols_with_na)} columns with remaining missing values"
            )

            # Store index for later reconstruction
            index = features.index

            # Apply imputation only to columns with missing values
            imputer = KNNImputer(n_neighbors=5)
            features[remaining_cols_with_na] = pd.DataFrame(
                imputer.fit_transform(features[remaining_cols_with_na]),
                columns=remaining_cols_with_na,
                index=index,
            )

    # Combine metadata and features
    filtered_data = pd.concat([metadata, features], axis=1).reset_index(drop=True)

    return filtered_data


def intensity_filter(
    cell_data, first_feature, channel_names=None, contamination=0.01
) -> pd.DataFrame:
    """Uses EllipticEnvelope to filter outliers by channel intensities.

    Derived from Recursion's EFAAR pipeline: https://github.com/recursionpharma/EFAAR_benchmarking/blob/60df3eb267de3ba13b95f720b2a68c85f6b63d14/efaar_benchmarking/efaar.py#L295

    Args:
        cell_data (pd.DataFrame): Cell data dataframe.
        first_feature (str): Name of the first feature column
        channel_names (list[str], optional): A list of channel names to use for intensity filtering. Defaults to None.
        contamination (float, optional): The proportion of outliers to expect. Defaults to 0.01.

    Returns:
        pd.DataFrame: Filtered cell data dataframe.
    """
    # Identify feature cols
    feature_start_idx = cell_data.columns.get_loc(first_feature)
    feature_cols = cell_data.columns[feature_start_idx:].tolist()

    # Determine intensity columns
    intensity_cols = [
        col
        for col in feature_cols
        if any(col.endswith(f"_{channel}_mean") for channel in channel_names)
    ]

    # Fit EllipticEnvelope to intensity cols and get mask
    mask = EllipticEnvelope(contamination=contamination, random_state=42).fit_predict(
        cell_data[intensity_cols]
    )

    # Return filtered cell data
    return cell_data[mask == 1].reset_index(drop=True)
