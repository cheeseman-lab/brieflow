"""Helper functions for evaluating the results of the evaluation step in the aggregate module.

This module includes functions for generating visualizations and testing data integrity.
It provides the following functionalities:
- Suggesting parameters for feature analysis based on input data.
- Visualization of feature distributions across datasets with violin plots.
- Detection and reporting of missing values, including NA, null, blank, and infinite values.
"""

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions(
    cell_data, feature_start_indx, num_features=5, num_samples=1000
):
    feature_data = cell_data.sample(num_samples)

    feature_col_indxs = random.sample(
        range(feature_start_indx, cell_data.shape[1]), num_features
    )
    feature_cols = cell_data.columns[feature_col_indxs]

    # Reshape data for seaborn (long format)
    plot_data = pd.DataFrame()
    for col in feature_cols:
        features = np.array(feature_data[col].tolist())
        temp_df = pd.DataFrame(
            {"Feature": features.flatten(), "Column": [col] * len(features.flatten())}
        )
        plot_data = pd.concat([plot_data, temp_df])

    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Column", y="Feature", data=plot_data)
    plt.title(f"Distribution of Features For {num_features} Columns")
    plt.tight_layout()
    plt.show()


def summarize_cell_data(
    cell_data: pd.DataFrame, classes: list, collapse_cols: list
) -> pd.DataFrame:
    """Summarizes cell data by counting total cells, class-specific cells, and unique metric values.

    Args:
        cell_data (pd.DataFrame): DataFrame containing cell metadata.
        classes (list): List of class names to filter.
        collapse_cols (list): List of column names to count unique values.

    Returns:
        pd.DataFrame: Summary table with stage names and corresponding counts.
    """
    counts = [("Raw Data", len(cell_data))]

    for class_name in classes:
        class_subset = cell_data[cell_data["class"] == class_name]
        counts.append((f"{class_name} Cells", len(class_subset)))

        for col in collapse_cols:
            counts.append((f"{class_name} {col}s", class_subset[col].nunique()))

    return pd.DataFrame(counts, columns=["Stage", "Count"])


def test_missing_values(gene_data, name):
    """Test for missing values in a dataframe, including NA, null, blank, and infinite values.

    Returns results in a format suitable for CSV export.

    Args:
        gene_data (pandas.DataFrame): DataFrame to test.
        name (str): Name of the dataset for printing.

    Returns:
        pandas.DataFrame: DataFrame containing detailed missing value information.
    """
    # Check for various types of missing or problematic values
    missing = pd.DataFrame(
        {
            "null_na": gene_data.isna().sum(),  # Catches np.nan, None, pd.NA
            "empty_string": gene_data.astype(str).eq("").sum(),  # Empty strings
            "infinite": gene_data.isin([np.inf, -np.inf]).sum(),  # Infinite values
        }
    )

    # Sum up all types of problematic values
    missing["total_issues"] = missing.sum(axis=1)

    # Calculate percentages
    missing["percentage"] = (missing["total_issues"] / len(gene_data) * 100).round(2)

    # Add dataset name
    missing["dataset"] = name

    # Add column names as a separate column
    missing["column"] = missing.index

    # Reorder columns
    missing = missing[
        [
            "dataset",
            "column",
            "total_issues",
            "percentage",
            "null_na",
            "empty_string",
            "infinite",
        ]
    ]

    # Filter for columns that have any issues
    results = missing[missing["total_issues"] > 0].reset_index(drop=True)

    # Print summary
    if not results.empty:
        for _, row in results.iterrows():
            details = []
            if row["null_na"] > 0:
                details.append(f"{row['null_na']} NA/null")
            if row["empty_string"] > 0:
                details.append(f"{row['empty_string']} empty")
            if row["infinite"] > 0:
                details.append(f"{row['infinite']} infinite")
    else:
        print(f"\nNo missing values found in {name} dataset")

    return results


def calculate_mitotic_percentage(df_mitotic, df_interphase):
    """Calculate the percentage of mitotic cells for each gene using pre-grouped data.

    Fills in zeros for missing genes in either dataset.

    Args:
        df_mitotic (DataFrame): DataFrame containing mitotic cell data (already grouped by gene).
        df_interphase (DataFrame): DataFrame containing interphase cell data (already grouped by gene).

    Returns:
        DataFrame: Contains gene names and their mitotic percentages.
    """
    # Get all unique genes from both datasets
    all_genes = sorted(
        list(set(df_mitotic["gene_symbol_0"]) | set(df_interphase["gene_symbol_0"]))
    )

    # Create dictionaries mapping genes to their counts
    mitotic_counts = dict(zip(df_mitotic["gene_symbol_0"], df_mitotic["gene_count"]))
    interphase_counts = dict(
        zip(df_interphase["gene_symbol_0"], df_interphase["gene_count"])
    )

    # Create result DataFrame with all genes, filling in zeros for missing values
    result_df = pd.DataFrame(
        {
            "gene": all_genes,
            "mitotic_cells": [mitotic_counts.get(gene, 0) for gene in all_genes],
            "interphase_cells": [interphase_counts.get(gene, 0) for gene in all_genes],
        }
    )

    # Report genes that were filled with zeros
    missing_in_mitotic = set(all_genes) - set(df_mitotic["gene_symbol_0"])
    missing_in_interphase = set(all_genes) - set(df_interphase["gene_symbol_0"])

    if missing_in_mitotic or missing_in_interphase:
        print("Note: Some genes were missing and filled with zero counts:")
        if missing_in_mitotic:
            print(
                f"Genes missing in mitotic data (filled with 0): {missing_in_mitotic}"
            )
        if missing_in_interphase:
            print(
                f"Genes missing in interphase data (filled with 0): {missing_in_interphase}"
            )

    # Calculate total cells and mitotic percentage
    result_df["total_cells"] = (
        result_df["mitotic_cells"] + result_df["interphase_cells"]
    )

    # Handle division by zero: if total_cells is 0, set percentage to 0
    result_df["mitotic_percentage"] = np.where(
        result_df["total_cells"] > 0,
        (result_df["mitotic_cells"] / result_df["total_cells"] * 100).round(2),
        0.0,
    )

    # Sort by mitotic percentage in descending order
    result_df = result_df.sort_values("mitotic_percentage", ascending=False)

    # Reset index to remove the old index
    result_df = result_df.reset_index(drop=True)

    # Print summary statistics
    print(f"\nProcessed {len(all_genes)} total genes")
    print(f"Average mitotic percentage: {result_df['mitotic_percentage'].mean():.2f}%")
    print(f"Median mitotic percentage: {result_df['mitotic_percentage'].median():.2f}%")

    return result_df
