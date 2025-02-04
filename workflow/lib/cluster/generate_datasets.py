def clean_and_validate(df):
    # Remove unnamed columns
    unnamed_cols = [col for col in df.columns if "Unnamed" in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Check for required columns
    required_cols = ["gene_symbol_0", "gene_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols}")

    # Reorder columns to ensure gene_count is after gene_symbol_0
    other_cols = [col for col in df.columns if col not in required_cols]
    new_cols = ["gene_symbol_0", "gene_count"] + other_cols
    df = df[new_cols]
    df = df.rename(columns={"gene_count": "cell_number"})

    return df


def split_channels(df, channel_combo, all_channels):
    """
    Filter dataframe to only include features from specified channel pair,
    removing features from other channels.

    Args:
        df (pd.DataFrame): Input dataframe with features
        channel_combo (list): Channels to keep.
        all_channels (list): List of all possible channels

    Returns:
        pd.DataFrame: Filtered dataframe with features only from specified channels
    """

    # Find channels to remove (those not in channel_combo)
    channels_to_remove = [ch for ch in all_channels if ch not in channel_combo]

    # Get all column names
    columns = df.columns.tolist()

    # Find columns to remove (those containing removed channel names)
    columns_to_remove = [
        col for col in columns if any(ch in col for ch in channels_to_remove)
    ]

    # Keep all columns except those from removed channels
    columns_to_keep = [col for col in columns if col not in columns_to_remove]

    return df[columns_to_keep]


def remove_low_number_genes(df, min_cells=10):
    """
    Remove genes with cell numbers below a certain threshold

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing 'cell_number' column
    min_cells : int, default=10
        Minimum number of cells required for a gene to be kept

    Returns:
    --------
    DataFrame
        DataFrame with genes filtered based on cell_number threshold
    """
    # Filter genes based on cell_number
    filtered_df = df[df["cell_number"] >= min_cells]

    # Print summary
    print("\nGene Filtering Summary:")
    print(f"Original genes: {len(df)}")
    print(f"Genes with < {min_cells} cells: {len(df) - len(filtered_df)}")
    print(f"Remaining genes: {len(filtered_df)}")

    return filtered_df


def remove_missing_features(df):
    """
    Remove features (columns) that contain any inf, nan, or blank values

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with features as columns

    Returns:
    --------
    DataFrame
        DataFrame with problematic features removed
    """
    import numpy as np

    df = df.copy()
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
