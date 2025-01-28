"""This module provides functions for processing and aggregating features during data analysis.

Functions include:
- Applying transformations to features using a flexible transformation dictionary.
- Suggesting parameters for feature analysis based on input data.
- Standardizing features using robust z-scores grouped by specified categories.
- Collapsing cell-level data to sgRNA-level and gene-level summaries with customizable options.

Functions:
    - feature_transform: Apply transformations to features based on a transformation dictionary.
    - suggest_parameters: Suggest feature analysis parameters from input data.
    - grouped_standardization: Perform robust z-score standardization grouped by specified features.
    - collapse_to_sgrna: Aggregate cell-level data to sgRNA-level summaries.
    - collapse_to_gene: Aggregate sgRNA-level data to gene-level summaries.
"""

import numpy as np
import pandas as pd


def feature_transform(df, transformation_dict, channels):
    """Apply transformations to features based on a transformation dictionary.

    Args:
        df (pd.DataFrame): Input dataframe containing features to transform.
        transformation_dict (pd.DataFrame): DataFrame containing 'feature' and 'transformation' columns specifying
            which transformations to apply to which features.
        channels (list): List of channel names to use when expanding feature templates.

    Returns:
        pd.DataFrame: DataFrame with transformed features.
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
    """Suggest parameters based on input dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        population_feature (str): Column name containing population identifiers.

    Returns:
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
    """Standardize features using robust z-scores, calculated per group using control populations.

    Args:
        df (pd.DataFrame): Input dataframe.
        population_feature (str): Column name containing population identifiers.
        control_prefix (str): Prefix identifying control populations.
        group_columns (list): Columns to group by for standardization.
        index_columns (list): Columns that uniquely identify cells.
        cat_columns (list): Categorical columns to preserve.
        target_features (list, optional): Features to standardize. If None, will standardize all numeric columns.
        drop_features (bool): Whether to drop untransformed features.

    Returns:
        pd.DataFrame: Standardized dataframe.
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


def collapse_to_sgrna(
    df,
    method="median",
    target_features=None,
    index_features=["gene_symbol_0", "sgRNA_0"],
    control_prefix="sg_nt",
    min_count=None,
):
    """Collapse cell-level data to sgRNA-level summaries.

    Args:
        df (pd.DataFrame): Input dataframe with cell-level data.
        method (str): Method for collapsing ('median' only currently supported).
        target_features (list, optional): Features to collapse. If None, uses all numeric columns.
        index_features (list): Columns that identify sgRNAs.
        control_prefix (str): Prefix identifying control sgRNAs.
        min_count (int, optional): Minimum number of cells required per sgRNA.

    Returns:
        pd.DataFrame: DataFrame with sgRNA-level summaries.
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    if method == "median":
        df_out = df.groupby(index_features)[target_features].median().reset_index()
        df_out["sgrna_count"] = (
            df.groupby(index_features)
            .size()
            .reset_index(name="sgrna_count")["sgrna_count"]
        )

        if min_count is not None:
            df_out = df_out.query("sgrna_count >= @min_count")
    else:
        raise ValueError("Only method='median' is currently supported")

    control_mask = df_out["gene_symbol_0"].str.startswith(control_prefix)
    unique_controls = df_out.loc[control_mask, "gene_symbol_0"].unique()
    if len(unique_controls) == 1:
        print("Multiple control guides not found. Renaming to ensure uniqueness.")
        control_mask = df_out["gene_symbol_0"].str.startswith(control_prefix)
        for idx, row_idx in enumerate(df_out[control_mask].index, 1):
            df_out.loc[row_idx, "gene_symbol_0"] = f"{control_prefix}_{idx}"

    return df_out


def collapse_to_gene(
    df, target_features=None, index_features=["gene_symbol_0"], min_count=None
):
    """Collapse sgRNA-level data to gene-level summaries.

    Args:
        df (pd.DataFrame): Input dataframe with sgRNA-level data.
        target_features (list, optional): Features to collapse. If None, uses all numeric columns.
        index_features (list): Columns that identify genes.
        min_count (int, optional): Minimum number of sgRNAs required per gene.

    Returns:
        pd.DataFrame: DataFrame with gene-level summaries.
    """
    if target_features is None:
        target_features = [col for col in df.columns if col not in index_features]

    df_out = df.groupby(index_features)[target_features].median().reset_index()

    if "sgrna_count" in df.columns:
        df_out["gene_count"] = (
            df.groupby(index_features)["sgrna_count"].sum().reset_index(drop=True)
        )

    if min_count is not None:
        df_out = df_out.query("gene_count >= @min_count")

    return df_out
