"""Module for generating and preprocessing datasets for clustering analysis.

This module provides functions to clean, filter, and validate datasets as part of the
clustering workflow. These functions ensure that the input data is properly formatted
and contains only valid features and relevant information for clustering processes.

Functions:
    - clean_and_validate: Clean and validate the input DataFrame by removing unnamed columns,
      ensuring required columns are present, and renaming/reordering columns.
    - split_channels: Filter the DataFrame to include features from specified channel pairs only.
    - remove_low_number_genes: Remove genes with a 'cell_number' below a given threshold.
    - remove_missing_features: Remove features containing any infinite, NaN, or blank values.
"""


def clean_and_validate(gene_data):
    """Clean gene data and validate required columns are present.

    This function removes unnamed columns, checks for required columns,
    reorders columns to ensure 'gene_count' is after 'gene_symbol_0', and
    renames 'gene_count' to 'cell_number'.

    Args:
        gene_data (pd.DataFrame): Input DataFrame to be cleaned and validated.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
    """
    # Remove unnamed columns
    unnamed_cols = [col for col in gene_data.columns if "Unnamed" in str(col)]
    if unnamed_cols:
        gene_data = gene_data.drop(columns=unnamed_cols)

    # Check for required columns
    required_cols = ["gene_symbol_0", "gene_count"]
    missing_cols = [col for col in required_cols if col not in gene_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols}")

    # Reorder columns to ensure gene_count is after gene_symbol_0
    other_cols = [col for col in gene_data.columns if col not in required_cols]
    new_cols = ["gene_symbol_0", "gene_count"] + other_cols
    gene_data = gene_data[new_cols]
    gene_data = gene_data.rename(columns={"gene_count": "cell_number"})

    return gene_data


def split_channels(gene_data, channel_combo, all_channels):
    """Filter dataframe to only include features from specified channel pair, removing features from other channels.

    Args:
        gene_data (pd.DataFrame): Input dataframe with features.
        channel_combo (list): Channels to keep.
        all_channels (list): List of all possible channels.

    Returns:
        pd.DataFrame: Filtered dataframe with features only from specified channels.
    """
    # Find channels to remove (those not in channel_combo)
    channels_to_remove = [ch for ch in all_channels if ch not in channel_combo]

    # Get all column names
    columns = gene_data.columns.tolist()

    # Find columns to remove (those containing removed channel names)
    columns_to_remove = [
        col for col in columns if any(ch in col for ch in channels_to_remove)
    ]

    # Keep all columns except those from removed channels
    columns_to_keep = [col for col in columns if col not in columns_to_remove]

    return gene_data[columns_to_keep]


def remove_low_number_genes(gene_data, min_cells=10):
    """Remove genes with cell numbers below a certain threshold.

    Args:
        gene_data (pd.DataFrame): Input DataFrame containing 'cell_number' column.
        min_cells (int, optional): Minimum number of cells required for a gene to be kept. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with genes filtered based on cell_number threshold.
    """
    # Filter genes based on cell_number
    filtered_df = gene_data[gene_data["cell_number"] >= min_cells]

    # Print summary
    print("\nGene Filtering Summary:")
    print(f"Original genes: {len(gene_data)}")
    print(f"Genes with < {min_cells} cells: {len(df) - len(filtered_df)}")
    print(f"Remaining genes: {len(filtered_df)}")

    return filtered_df


def remove_missing_features(gene_data):
    """Remove features (columns) that contain any inf, nan, or blank values.

    Args:
        gene_data (pd.DataFrame): Input DataFrame with features as columns.

    Returns:
        pd.DataFrame: DataFrame with problematic features removed.
    """
    import numpy as np

    df = gene_data.copy()
    removed_features = {}

    # Check for infinite values
    inf_features = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    if inf_features:
        removed_features["infinite"] = inf_features
        df = df.drop(columns=inf_features)

    # Check for null/na values
    null_features = df.columns[df.isna().any()].tolist()
    if null_features:
        removed_features["null_na"] = null_features
        df = df.drop(columns=null_features)

    # Check for empty strings (for string columns only)
    string_cols = df.select_dtypes(include=["object"]).columns
    if len(string_cols) > 0:
        empty_features = string_cols[df[string_cols].astype(str).eq("").any()].tolist()
        if empty_features:
            removed_features["empty_string"] = empty_features
            df = df.drop(columns=empty_features)

    # Print summary
    print("\nFeature Cleaning Summary:")
    print(
        f"Original features: {len(df.columns) + sum(len(v) for v in removed_features.values())}"
    )

    if removed_features:
        print("\nRemoved features:")
        if "infinite" in removed_features:
            print(
                f"\nFeatures with infinite values ({len(removed_features['infinite'])}):"
            )
            for feat in removed_features["infinite"]:
                print(f"- {feat}")

        if "null_na" in removed_features:
            print(
                f"\nFeatures with null/NA values ({len(removed_features['null_na'])}):"
            )
            for feat in removed_features["null_na"]:
                print(f"- {feat}")

        if "empty_string" in removed_features:
            print(
                f"\nFeatures with empty strings ({len(removed_features['empty_string'])}):"
            )
            for feat in removed_features["empty_string"]:
                print(f"- {feat}")
    else:
        print("\nNo problematic features found!")

    print(f"\nRemaining features: {len(df.columns)}")

    return df
