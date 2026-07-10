"""This module provides functions for filtering cell data during aggregation.

Available filters:
- query_filter: Apply pandas query strings to filter cells
- perturbation_filter: Remove cells without perturbation assignments
- missing_values_filter: Handle missing values through dropping or imputation
- intensity_filter: Remove outliers based on channel intensities using LocalOutlierFactor
- harmonize_pool_schema: Compute a consistent feature-column set for a pool of per-well parquets
"""

import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor


def harmonize_pool_schema(
    paths: list[str],
    metadata_cols: list[str],
    drop_cols_threshold: float | None = None,
) -> tuple[list[str], list[str], dict]:
    """Compute a consistent (metadata, feature) column set for a multi-file pool.

    When per-well filter decisions diverge — e.g. missing_values_filter drops
    different columns in different wells because class composition varies — naive
    pool reads via pyarrow's dataset union produce NaN blocks that break
    downstream operations (PCA, center-scale). This helper produces a single
    harmonized column set to use across every scan of the pool, matching the
    drop-column convention from missing_values_filter.

    Steps:
      1. Schema intersection across `paths`. Columns present in some but not all
         files are "lost to per-well schema mismatch" (typically driven by
         class-composition differences across wells) and logged per-file.
      2. If `drop_cols_threshold` is provided, scan the intersected pool once and
         drop any feature column whose pool-level NaN proportion is >= threshold.
         Uses the same comparison as missing_values_filter.
    Row-level NaN cleanup (drop_rows_threshold, impute) is intentionally
    deferred to the caller so it can be applied on each working subset.

    Args:
        paths (list[str]): parquet file paths forming the pool.
        metadata_cols (list[str]): canonical metadata column names (only those
            present in every file are returned).
        drop_cols_threshold (float | None): pool-level NaN proportion threshold
            above which a feature column is dropped. None disables this step.

    Returns:
        tuple:
            kept_metadata_cols (list[str]),
            kept_feature_cols (list[str]),
            report (dict) — keys: "schema_mismatch" (dict file -> missing cols),
                                  "threshold_dropped" (list of col names).
    """
    if not paths:
        return [], [], {"schema_mismatch": {}, "threshold_dropped": []}

    per_file_schemas = {p: set(pq.read_schema(p).names) for p in paths}
    union_cols = set().union(*per_file_schemas.values())
    intersection_cols = set.intersection(*per_file_schemas.values())
    schema_mismatch = {
        p: sorted(union_cols - cols)
        for p, cols in per_file_schemas.items()
        if union_cols - cols
    }

    if schema_mismatch:
        total_lost = len(union_cols - intersection_cols)
        print(
            f"[pool] dropping {total_lost} column(s) present in some but not all "
            f"of {len(paths)} input files (per-file missingness — typically driven "
            f"by class composition differences across wells):"
        )
        for p, missing in schema_mismatch.items():
            preview = ", ".join(missing[:5]) + (
                f", ...(+{len(missing) - 5} more)" if len(missing) > 5 else ""
            )
            print(f"  {p.split('/')[-1]}: missing {len(missing)} col(s): {preview}")

    # Preserve caller-provided order within metadata_cols; feature order is sorted
    # for determinism since there's no natural ordering after an intersection.
    kept_metadata_cols = [c for c in metadata_cols if c in intersection_cols]
    kept_feature_cols = sorted(intersection_cols - set(metadata_cols))

    threshold_dropped = []
    if drop_cols_threshold is not None and kept_feature_cols:
        # Pin to one reference schema so pyarrow casts each file to it instead of
        # auto-unifying (which raises ArrowInvalid when files share a column with a
        # mismatched dtype). paths are non-empty and share the kept columns.
        pool = ds.dataset(
            paths, format="parquet", schema=pq.read_schema(paths[0])
        ).to_table(columns=kept_feature_cols)
        # Compute pool-level NaN proportion per column without materializing full pandas.
        total_rows = pool.num_rows
        if total_rows > 0:
            for col_name in list(kept_feature_cols):
                nulls = pool.column(col_name).null_count
                if nulls / total_rows >= drop_cols_threshold:
                    threshold_dropped.append(col_name)
        if threshold_dropped:
            print(
                f"[pool] dropping {len(threshold_dropped)} column(s) with pool-level "
                f"NaN proportion >= {drop_cols_threshold * 100:.1f}%: "
                f"{', '.join(threshold_dropped[:10])}"
                f"{', ...' if len(threshold_dropped) > 10 else ''}"
            )
            kept_feature_cols = [
                c for c in kept_feature_cols if c not in threshold_dropped
            ]

    return (
        kept_metadata_cols,
        kept_feature_cols,
        {
            "schema_mismatch": schema_mismatch,
            "threshold_dropped": threshold_dropped,
        },
    )


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

            # Cast integer columns to float64 before imputation (KNNImputer returns float64)
            for col in remaining_cols_with_na:
                if pd.api.types.is_integer_dtype(features[col]):
                    features[col] = features[col].astype("float64")

            # Use positional (iloc) indexing throughout — features.index may have
            # duplicate labels (e.g. when concat across wells doesn't reset),
            # which breaks .loc-based reads/writes.
            has_na_mask = features[remaining_cols_with_na].isna().any(axis=1).to_numpy()
            na_positions = np.flatnonzero(has_na_mask)
            non_na_positions = np.flatnonzero(~has_na_mask)
            col_positions = [
                features.columns.get_loc(c) for c in remaining_cols_with_na
            ]

            np.random.seed(42)
            for i in range(0, len(na_positions), batch_size):
                batch_pos = na_positions[i : i + batch_size]
                print(
                    f"Imputing for batch {i // batch_size + 1} with {len(batch_pos)} NA rows"
                )

                sampled_non_na_pos = np.random.choice(
                    non_na_positions,
                    size=min(sample_size, len(non_na_positions)),
                    replace=False,
                )
                combined_pos = np.concatenate([batch_pos, sampled_non_na_pos])

                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(
                    features.iloc[combined_pos, col_positions].to_numpy()
                )

                features.iloc[batch_pos, col_positions] = imputed_values[
                    : len(batch_pos)
                ]

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
    # Handle contamination parameter validation
    if contamination < 0 or contamination >= 0.5:
        raise ValueError(
            "Contamination must be between 0 (inclusive) and 0.5 (exclusive)"
        )

    if contamination == 0:
        print("Contamination is 0, skipping intensity filtering")
        return metadata, features

    if len(metadata) == 0:
        print("No cells to filter, skipping intensity filtering")
        return metadata, features

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
