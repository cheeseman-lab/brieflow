"""This module provides functions for loading and formatting data during aggregation.

Functions include:
- Loading a subset of data from parquet files for efficient processing.

Functions:
    - load_parquet_subset: Load a fixed number of random rows from a parquet file.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor


def query_filter(metadata, features, queries):
    """Sequentially apply a list of query strings to filter metadata and features DataFrames.

    Example: queries=["mapped_single_gene == False", "cell_quality_score > 0.8"] filters for unmapped single-gene cells with high quality.

    Args:
        metadata: The input metadata DataFrame.
        features: The input features DataFrame.
        queries: List of query strings to apply using DataFrame.query().

    Returns:
        A tuple of (filtered_metadata, filtered_features).
    """
    if queries is None:
        return metadata, features

    for q in queries:
        before = len(metadata)
        # Apply query filter to metadata
        metadata_filtered = metadata.query(q)
        # Filter features to match metadata indices
        features_filtered = features.loc[metadata_filtered.index]
        after = len(metadata_filtered)
        print(f"Query '{q}' filtered out {before - after} cells")
        metadata, features = metadata_filtered, features_filtered

    return metadata.reset_index(drop=True), features.reset_index(drop=True)


def perturbation_filter(
    metadata,
    features,
    perturbation_name_col,
):
    """Clean cell data by removing cells without perturbation assignments.

    Args:
        metadata: DataFrame containing metadata.
        features: DataFrame containing features.
        perturbation_name_col: Column name in metadata containing perturbation assignments.

    Returns:
        tuple: (filtered_metadata, filtered_features)
    """
    # Get indices of cells with perturbation assignments
    valid_indices = metadata[metadata[perturbation_name_col].notna()].index

    # Filter both metadata and features
    filtered_metadata = metadata.loc[valid_indices]
    filtered_features = features.loc[valid_indices]

    print(f"Found {len(filtered_metadata)} cells with assigned perturbations")

    return filtered_metadata.reset_index(drop=True), filtered_features.reset_index(
        drop=True
    )


def missing_values_filter(
    metadata,
    features,
    drop_cols_threshold=None,
    drop_rows_threshold=None,
    impute=False,
    batch_size=1000,
    sample_size=10000,
):
    """Filter cell data by handling missing values through dropping or batched imputation.

    Args:
        metadata (pd.DataFrame): DataFrame containing metadata.
        features (pd.DataFrame): DataFrame containing features.
        drop_cols_threshold (float, optional): If provided, drops columns with NaN proportion >= threshold.
                                              Range: 0.0-1.0. Defaults to None.
        drop_rows_threshold (float, optional): If provided, drops rows with NaN proportion >= threshold.
                                              Range: 0.0-1.0. Defaults to None.
        impute (bool): Whether to impute remaining missing values after dropping. Defaults to False.
        batch_size (int): Number of NA rows to process in each batch. Defaults to 1000.
        sample_size (int): Number of non-NA rows to sample for each batch. Defaults to 10000.

    Returns:
        tuple: (filtered_metadata, filtered_features)
    """
    # Get columns with missing values
    cols_with_na = features.columns[features.isna().any()].tolist()

    if not cols_with_na:
        return metadata, features

    # Handle column dropping based on threshold
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

    # Handle row dropping based on threshold
    if drop_rows_threshold is not None:
        # Calculate proportion of NaN values in each row
        row_na_proportions = features.isna().sum(axis=1) / features.shape[1]

        # Identify rows to keep (inverse of rows to drop)
        rows_to_keep = row_na_proportions < drop_rows_threshold

        original_row_count = features.shape[0]
        features = features.loc[rows_to_keep]
        metadata = metadata.loc[rows_to_keep]

        print(
            f"Dropped {original_row_count - features.shape[0]} rows with ≥{drop_rows_threshold * 100}% missing values"
        )

    # Impute remaining missing values if requested
    if impute:
        # Get updated list of columns with missing values
        remaining_cols_with_na = features.columns[features.isna().any()].tolist()

        if remaining_cols_with_na:
            print(
                f"Imputing {len(remaining_cols_with_na)} columns with remaining missing values using batched KNN"
            )

            # Identify rows with any NAs in the remaining columns
            has_na_mask = features[remaining_cols_with_na].isna().any(axis=1)
            na_rows_idx = features.index[has_na_mask]
            non_na_rows_idx = features.index[~has_na_mask]

            np.random.seed(42)
            # Process NA rows in batches
            for i in range(0, len(na_rows_idx), batch_size):
                batch_na_idx = na_rows_idx[i : i + batch_size]
                print(
                    f"Imputing for batch {i // batch_size + 1} with {len(batch_na_idx)} NA rows"
                )

                # Sample non-NA rows randomly instead of stratified sampling
                sampled_non_na_idx = np.random.choice(
                    non_na_rows_idx,
                    size=min(sample_size, len(non_na_rows_idx)),
                    replace=False,
                )

                # Combine sampled non-NA rows with current batch of NA rows
                batch_idx = np.concatenate([batch_na_idx, sampled_non_na_idx])

                # Perform KNN imputation on this batch
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(
                    features.loc[batch_idx, remaining_cols_with_na]
                )

                # Update only the NA rows with imputed values
                na_rows_in_batch = np.arange(len(batch_na_idx))
                features.loc[batch_na_idx, remaining_cols_with_na] = pd.DataFrame(
                    imputed_values[na_rows_in_batch],
                    index=batch_na_idx,
                    columns=remaining_cols_with_na,
                )

    return metadata, features


def intensity_filter(
    metadata, features, channel_names=None, contamination=0.01
) -> pd.DataFrame:
    """Uses LocalOutlierFactor to filter outliers by channel intensities.

    Derived from Recursion's EFAAR pipeline: https://github.com/recursionpharma/EFAAR_benchmarking/blob/60df3eb267de3ba13b95f720b2a68c85f6b63d14/efaar_benchmarking/efaar.py#L295

    Args:
        metadata (pd.DataFrame): DataFrame containing metadata.
        features (pd.DataFrame): DataFrame containing features.
        channel_names (list[str], optional): A list of channel names to use for intensity filtering. Defaults to None.
        contamination (float, optional): The proportion of outliers to expect. Defaults to 0.01.

    Returns:
        tuple: (filtered_metadata, filtered_features) DataFrames with outliers removed.
    """
    # Identify feature cols
    feature_cols = features.columns.tolist()

    # Determine intensity columns
    intensity_cols = [
        col
        for col in feature_cols
        if any(col.endswith(f"_{channel}_mean") for channel in channel_names)
    ]

    # Fit LocalOutlierFactor to intensity cols and get mask
    mask = LocalOutlierFactor(
        contamination=contamination,
        n_jobs=-1,
    ).fit_predict(features[intensity_cols])

    # Return filtered cell data
    return metadata[mask == 1].reset_index(drop=True), features[mask == 1].reset_index(
        drop=True
    )
