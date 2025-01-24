import numpy as np
import pandas as pd


def feature_transform(df, transformation_dict, channels):
    """
    Apply transformations to features based on a transformation dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features to transform
    transformation_dict : pd.DataFrame
        DataFrame containing 'feature' and 'transformation' columns specifying
        which transformations to apply to which features
    channels : list
        List of channel names to use when expanding feature templates

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed features
    """

    def apply_transformation(feature, transformation):
        if transformation == "log(feature)":
            return np.log(feature)
        elif transformation == "log(feature-1)":
            return np.log(feature - 1)
        elif transformation == "log(1-feature)":
            return np.log(1 - feature)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    df = df.copy()

    for _, row in transformation_dict.iterrows():
        feature_template = row["feature"]
        transformation = row["transformation"]

        # Handle single channel features
        if "{channel}" in feature_template:
            for channel in channels:
                feature = feature_template.replace("{channel}", channel)
                if feature in df.columns:
                    df[feature] = apply_transformation(df[feature], transformation)

        # Handle double channel features (overlap)
        elif "{channel1}" in feature_template and "{channel2}" in feature_template:
            for channel1 in channels:
                for channel2 in channels:
                    if channel1 != channel2:
                        feature = feature_template.replace(
                            "{channel1}", channel1
                        ).replace("{channel2}", channel2)
                        if feature in df.columns:
                            df[feature] = apply_transformation(
                                df[feature], transformation
                            )

    return df


def suggest_parameters(df, population_feature):
    """
    Suggest parameters based on input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    population_feature : str
        Column name containing population identifiers

    Returns
    -------
    None
    """

    # Look for potential control prefixes
    unique_populations = df[population_feature].unique()
    potential_controls = [
        pop
        for pop in unique_populations
        if any(
            control in pop.lower()
            for control in ["nt", "non-targeting", "control", "ctrl", "neg"]
        )
    ]

    # Find first feature-like column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    potential_features = [
        col
        for col in numeric_cols
        if any(
            pattern in col.lower()
            for pattern in ["mean", "median", "std", "intensity", "area"]
        )
    ]

    # Identify metadata-like columns
    potential_metadata = df.select_dtypes(include=["object"]).columns.tolist()

    print("Suggested Parameters:")
    print("-" * 50)

    if potential_controls:
        print("\nPotential control prefixes found:")
        for ctrl in potential_controls:
            print(f"  - '{ctrl}'")
    else:
        print("\nNo obvious control prefixes found. Please check your data.")

    if potential_features:
        print(f"\nFirst few feature columns detected:")
        for feat in potential_features[:5]:
            print(f"  - '{feat}'")

    print("\nMetadata columns detected:")
    print(f"  - Categorical: {', '.join(potential_metadata[:5])}")


def grouped_standardization(
    df,
    population_feature="gene_symbol_0",
    control_prefix="sg_nt",
    group_columns=["well"],
    index_columns=["tile", "cell_0"],
    cat_columns=["gene_symbol_0", "sgRNA_0"],
    target_features=None,
    drop_features=False,
):
    """
    Standardize features using robust z-scores, calculated per group using control populations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    population_feature : str
        Column name containing population identifiers
    control_prefix : str
        Prefix identifying control populations
    group_columns : list
        Columns to group by for standardization
    index_columns : list
        Columns that uniquely identify cells
    cat_columns : list
        Categorical columns to preserve
    target_features : list, optional
        Features to standardize. If None, will standardize all numeric columns
    drop_features : bool
        Whether to drop untransformed features

    Returns
    -------
    pd.DataFrame
        Standardized dataframe
    """
    df_out = df.copy().drop_duplicates(subset=group_columns + index_columns)

    if target_features is None:
        target_features = [
            col
            for col in df.columns
            if col not in group_columns + index_columns + cat_columns
        ]

    if drop_features:
        df = df[group_columns + index_columns + cat_columns + target_features]

    unstandardized_features = [col for col in df.columns if col not in target_features]

    # Filter control group
    control_group = df[df[population_feature].str.startswith(control_prefix)]

    # Calculate MAD
    def median_absolute_deviation(arr):
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return mad

    # Calculate group statistics
    group_medians = control_group.groupby(group_columns)[target_features].median()
    group_mads = control_group.groupby(group_columns)[target_features].apply(
        lambda x: x.apply(median_absolute_deviation)
    )

    # Standardize using robust z-score
    df_out = pd.concat(
        [
            df_out[unstandardized_features].set_index(group_columns + index_columns),
            df_out.set_index(group_columns + index_columns)[target_features]
            .subtract(group_medians)
            .divide(group_mads)
            .multiply(0.6745),  # Scale factor for robust z-score
        ],
        axis=1,
    )

    return df_out.reset_index()
