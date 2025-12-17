"""This module provides functions for training cell classifiers using various ML algorithms.

Includes utilities for data loading, feature selection, model training with grid search,
evaluation metrics, and visualization of results. The main entry point is train_classifier_pipeline
which trains multiple model configurations and saves the best performers.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from lib.aggregate.cell_classification import CellClassifier
from lib.aggregate.cell_data_utils import split_cell_data


def setup_publication_plot_style():
    """Set up a consistent plotting style for publication-quality figures.

    Args:
        None
    Returns:
        cell_classifier_cmap: A custom colormap for cell classification plots.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "Bitstream Vera Sans",
                "sans-serif",
            ],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "figure.figsize": (10, 8),
            "figure.dpi": 300,
        }
    )

    # Create a distinct color palette different from the benchmark scripts
    # Using a blue to purple to red custom colormap without trying to register it
    colors = [
        (0.0, 0.4, 0.8),
        (0.6, 0.0, 0.8),
        (0.8, 0.0, 0.4),
    ]  # Blue to purple to red
    cell_classifier_cmap = LinearSegmentedColormap.from_list(
        "cell_classifier", colors, N=100
    )

    return cell_classifier_cmap


def create_run_directories(
    output_root_dir: str | None = None, training_name: str | None = None
):
    """Create run-specific output directories under a given root.

    Args:
        output_root_dir: Root directory where outputs will be created. If None, defaults to 'analysis_root'.
        training_name: Optional name to append to the base classifier folder. If provided,
        base folder becomes 'classifier_<training_name>', otherwise 'classifier'.

    Returns:
        a dict with paths for: base, run, statistics, models, plots, results, timestamp.
    """
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    # Determine root and base folder name
    if output_root_dir is None:
        output_root_dir = "analysis_root"

    base_folder = f"classifier_{training_name}" if training_name else "classifier"

    # Create base output directories
    base_output_dir = os.path.join(output_root_dir, base_folder)
    run_dir = os.path.join(base_output_dir, run_name)

    # Create subdirectories
    statistics_dir = os.path.join(run_dir, "statistics")
    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    results_dir = os.path.join(run_dir, "results")

    for directory in [statistics_dir, models_dir, plots_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"Created output directories for run: {run_name} under {base_output_dir}")

    return {
        "base": base_output_dir,
        "run": run_dir,
        "statistics": statistics_dir,
        "models": models_dir,
        "plots": plots_dir,
        "results": results_dir,
        "timestamp": timestamp,
    }


def load_cellprofiler_data(file_paths, class_title=None, metadata_cols=None):
    """Load and combine multiple CellProfiler parquet files.

    Args:
        file_paths: List of parquet file paths to load.
        class_title: Optional str. Name of the classification column to validate exists in data.
        metadata_cols: Optional list. Metadata columns that class_title should be present in.

    Returns:
        Combined DataFrame with all data from the provided parquet files.

    Raises:
        ValueError: If class_title is specified but not found in loaded data.
        ValueError: If class_title is provided but not in metadata_cols.
    """
    all_data = []

    for file_path in file_paths:
        # Load data
        data = pd.read_parquet(file_path)
        all_data.append(data)

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    # Reset index
    combined_data.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(combined_data)} cells from {len(file_paths)} files")

    # Validate class_title exists in data if provided
    if class_title is not None:
        if class_title not in combined_data.columns:
            available_class_cols = [
                c for c in combined_data.columns
                if c not in ['plate', 'well', 'tile', 'label']
                and not c.startswith(('nucleus_', 'cell_', 'cytoplasm_', 'vacuole_'))
            ]
            raise ValueError(
                f"CLASS_TITLE '{class_title}' not found in training data.\n"
                f"Available classification columns: {available_class_cols}"
            )

        # Validate class_title is in metadata_cols if provided
        if metadata_cols is not None and class_title not in metadata_cols:
            raise ValueError(
                f"CLASS_TITLE '{class_title}' must be included in METADATA_COLS.\n"
                f"Current METADATA_COLS: {metadata_cols}"
            )

    return combined_data


def apply_cell_category_mapping(
    data,
    label_col,
    remove_phases=None,
    output_col="phase",
    category_col="category",
    mapping=None,
    default_category="Unknown",
    verbose=True,
):
    """Map raw cell cycle labels to standardized phases and categories.

    Args:
        data: pd.DataFrame DataFrame containing cell metadata with raw labels.
        label_col: str Column name in data with raw cell cycle labels.
        remove_phases: list or None List of phases to remove from the data after mapping.
        output_col: str Column name to store mapped phase labels.
        category_col: str Column name to store mapped category labels.
        mapping: dict or None Mapping dictionary with 'label_to_class' and 'class
        default_category: str Category label to assign for unmapped or unknown phases.
        verbose: bool Whether to print summary information.

    Returns:
        pd.DataFrame with added phase and category columns, and optionally filtered.
    """
    # Create a copy to avoid modifying the original
    result_df = data.copy()

    # Check if label column exists
    if label_col not in result_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Apply default mapping for cell cycle if none provided
    if mapping is None:
        raise ValueError("Mapping dictionary is required")

    # Ensure all required mapping keys exist
    required_keys = ["label_to_class", "class_to_category"]
    missing_keys = [key for key in required_keys if key not in mapping]
    if missing_keys:
        raise ValueError(f"Missing required mapping keys: {missing_keys}")

    # Apply label to phase mapping
    phase_mapping = mapping["label_to_class"]
    result_df[output_col] = result_df[label_col].map(
        lambda x: phase_mapping.get(x, default_category)
    )

    # Apply phase to category mapping
    category_mapping = mapping["class_to_category"]
    result_df[category_col] = result_df[output_col].map(
        lambda x: category_mapping.get(x, default_category)
    )

    # Store phase and category colors if provided
    if "phase_colors" in mapping:
        # Store as attribute for later use in plotting
        result_df.attrs["phase_colors"] = mapping["phase_colors"]

    if "category_colors" in mapping:
        # Store as attribute for later use in plotting
        result_df.attrs["category_colors"] = mapping["category_colors"]

    # Print summary if verbose
    if verbose:
        print("\nCategory Mapping Summary:")
        print(
            f"Original labels in {label_col}: {sorted(result_df[label_col].unique())}"
        )
        print(
            f"Mapped phases in {output_col}: {sorted(result_df[output_col].unique())}"
        )
        print(
            f"Mapped categories in {category_col}: {sorted(result_df[category_col].unique())}"
        )

        # Display counts for phases and categories
        print("\nPhase Distribution:")
        phase_counts = result_df[output_col].value_counts()
        for phase, count in phase_counts.items():
            percentage = count / len(result_df) * 100
            print(f"  {phase}: {count} cells ({percentage:.1f}%)")

        print("\nCategory Distribution:")
        category_counts = result_df[category_col].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(result_df) * 100
            print(f"  {category}: {count} cells ({percentage:.1f}%)")

    # Remove unwanted phases if specified
    if remove_phases:
        original_count = len(result_df)
        result_df = result_df[~result_df[output_col].isin(remove_phases)]
        removed_count = original_count - len(result_df)

        if verbose and removed_count > 0:
            print(f"\nRemoved {removed_count} cells with phases: {remove_phases}")
            print(f"Remaining cells: {len(result_df)}")

    return result_df


def select_features_from_split(
    features_df,
    feature_markers=None,
    exclude_markers=None,
    exclude_cols=None,
    remove_nan=True,
    verbose=True,
):
    """Select features from the features dataframe.

    Args:
    features_df : pd.DataFrame Features dataframe from split_cell_data
    feature_markers : dict Dictionary of marker names to include mapped to True/False
        Example: {'DAPI': True, 'ACTIN': False}
    exclude_markers : list List of marker strings to exclude from features
    exclude_cols : list
        List of specific columns to exclude from features
    remove_nan : bool
        Whether to remove columns with NaN values
    verbose : bool
        Whether to print details about selected features
    Returns:
    list List of selected feature column names
    """
    # Default values
    if feature_markers is None:
        feature_markers = {"DAPI": True}  # By default, only include DAPI features

    if exclude_markers is None:
        exclude_markers = []
        if verbose:
            print("No markers to exclude")

    if exclude_cols is None:
        exclude_cols = []
        if verbose:
            print("No specific columns to exclude")

    # Get all columns as potential features
    all_feature_cols = features_df.columns.tolist()

    # Remove explicitly excluded columns
    feature_cols = [
        col
        for col in all_feature_cols
        if col not in exclude_cols
        and not any(col.startswith(ex) for ex in exclude_cols)
    ]

    # Initialize feature lists by marker
    feature_sets = {marker: [] for marker in feature_markers}
    feature_sets["morphology"] = []  # For features not associated with any marker

    # Categorize features by marker
    for col in feature_cols:
        # Check if column belongs to any marker category
        assigned = False
        for marker in feature_markers:
            if marker in col and not any(ex in col for ex in exclude_markers):
                feature_sets[marker].append(col)
                assigned = True
                break

        # If not assigned to any marker, it's a morphology feature
        if not assigned and not any(ex in col for ex in exclude_markers):
            feature_sets["morphology"].append(col)

    # Combine selected feature sets based on user's choices
    selected_features = []
    for marker, include in feature_markers.items():
        if include and marker in feature_sets:
            selected_features.extend(feature_sets[marker])

    # Always include morphology features unless explicitly turned off
    if feature_markers.get("morphology", True):
        selected_features.extend(feature_sets["morphology"])

    # Remove duplicates and sort for consistency
    selected_features = list(set(selected_features))
    selected_features.sort()

    # Remove columns with NaN values if requested
    if remove_nan:
        if verbose:
            print(
                f"Number of selected features before NaN removal: {len(selected_features)}"
            )
        selected_features = [
            col for col in selected_features if features_df[col].notna().all()
        ]
        if verbose:
            print(
                f"Number of selected features after NaN removal: {len(selected_features)}"
            )

    # Print feature information if verbose
    if verbose:
        print("\nFeature Selection Summary:")
        print(f"Total features selected: {len(selected_features)}")
        print("\nFeatures by category:")
        for category, features in feature_sets.items():
            if feature_markers.get(category, category == "morphology"):
                included = [f for f in features if f in selected_features]
                print(f"  {category}: {len(included)} features")
                if len(included) > 0 and len(included) <= 5:
                    print(f"    Examples: {', '.join(included)}")
                elif len(included) > 5:
                    print(f"    Examples: {', '.join(included[:5])}...")

    return selected_features


def plot_distribution_statistics(
    data,
    target_col,
    split_col=None,
    output_dir=".",
    prefix="",
    exclude_values=None,
    target_order=None,
    class_mapping=None,
    figsize=(10, 7),
    palette=None,
    dpi=300,
):
    """Plot distribution statistics for a target variable, optionally split by another variable.

    Args:
        data: pd.DataFrame DataFrame containing the data.
        target_col: str Column name of the target variable to analyze.
        split_col: str or None Column name to split the analysis by (e.g., batch). If None, no split.
        output_dir: str Directory to save plots.
        prefix: str Prefix for saved plot filenames.
        exclude_values: list or None List of target values to exclude from analysis.
        target_order: list or None List defining the order of target values for plotting.
        class_mapping: dict or None Optional mapping of class labels to display names.
        figsize: tuple Figure size for plots.
        palette: str, list, or None Color palette for plots. If 'cell_classifier', uses custom cmap.
        dpi: int DPI for saved figures.

    Returns:
        dict Summary statistics about the target variable distribution.
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    if split_col and split_col not in data.columns:
        raise ValueError(f"Split column '{split_col}' not found in data")

    os.makedirs(output_dir, exist_ok=True)

    # Handle palette
    if palette == "cell_classifier":
        palette = None

    # Filter unwanted values
    filtered = data
    if exclude_values:
        filtered = data[~data[target_col].isin(exclude_values)]

    # Build mapping (numeric -> display string) if provided, else infer from target_order for numeric targets
    label_to_name = None
    if (
        class_mapping
        and isinstance(class_mapping, dict)
        and "label_to_class" in class_mapping
    ):
        # Preserve insertion order from mapping
        label_to_name = dict(class_mapping["label_to_class"])
        display_order = list(label_to_name.values())
    else:
        # Fallback: if target_order is provided as list of strings and target is numeric, zip by sorted numeric ids
        col_is_numeric = pd.api.types.is_numeric_dtype(filtered[target_col])
        if (
            target_order
            and all(isinstance(x, str) for x in target_order)
            and col_is_numeric
        ):
            unique_ids = sorted(filtered[target_col].unique().tolist())
            if len(unique_ids) == len(target_order):
                label_to_name = {cid: nm for cid, nm in zip(unique_ids, target_order)}
                display_order = target_order
            else:
                label_to_name = None
                display_order = None
        else:
            display_order = (
                target_order  # could be None or a list matching actual values
            )

    # Create a display series for plotting
    if label_to_name is not None:
        display_series = (
            filtered[target_col]
            .map(label_to_name)
            .fillna(filtered[target_col].astype(str))
        )
    else:
        display_series = filtered[target_col].astype(str)

    display_col = f"{target_col}__display"
    filtered = filtered.copy()
    filtered[display_col] = display_series

    # Determine final order of displayed categories
    if display_order is None:
        target_values = [v for v in filtered[display_col].unique().tolist()]
    else:
        # Keep only those present
        present = set(filtered[display_col].unique())
        target_values = [v for v in display_order if v in present]
        # Add any missing at end (unlikely)
        for v in filtered[display_col].unique():
            if v not in target_values:
                target_values.append(v)

    # 1) Overall distribution (by display strings)
    plt.figure(figsize=figsize)
    ax = sns.countplot(
        data=filtered,
        x=display_col,
        hue=display_col,
        order=target_values,
        hue_order=target_values,
        palette=palette,
        legend=False,
    )
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title(f"{target_col} Distribution", fontsize=16, fontweight="bold")
    plt.xlabel(target_col, fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}{target_col}_distribution.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    # 2) Distribution by split variable
    if split_col:
        plt.figure(figsize=figsize)
        split_counts = (
            filtered.groupby([split_col, display_col]).size().unstack(fill_value=0)
        )
        split_percentages = split_counts.div(split_counts.sum(axis=1), axis=0) * 100
        # Reorder columns to target_values if present
        cols_in_order = [c for c in target_values if c in split_percentages.columns]
        split_percentages = split_percentages[cols_in_order]
        ax = split_percentages.plot(
            kind="bar",
            stacked=True,
            figsize=figsize,
            colormap=palette if isinstance(palette, str) else None,
            color=palette if not isinstance(palette, str) else None,
        )
        plt.title(
            f"{target_col} Distribution by {split_col}", fontsize=16, fontweight="bold"
        )
        plt.xlabel(split_col, fontsize=14)
        plt.ylabel("Percentage (%)", fontsize=14)
        plt.xticks(rotation=0)
        leg = plt.legend(
            title=target_col,
            title_fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            fontsize=10,
        )
        leg.get_frame().set_edgecolor("lightgray")
        for i, bar in enumerate(ax.patches):
            if bar.get_height() > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_height():.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}{target_col}_by_{split_col}.png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        # Per-class bar by split
        for target_value in target_values:
            value_percent = (
                filtered.groupby(split_col)[display_col]
                .apply(lambda x: sum(x == target_value) / len(x) * 100)
                .reset_index()
            )
            value_percent.columns = [split_col, f"{target_value} Percentage (%)"]
            plt.figure(figsize=figsize)
            ax = sns.barplot(
                x=split_col,
                y=f"{target_value} Percentage (%)",
                data=value_percent,
                color="#3498db",
                alpha=0.8,
            )
            for i, v in enumerate(value_percent[f"{target_value} Percentage (%)"]):
                ax.text(
                    i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold"
                )
            plt.title(
                f"{target_value} Percentage by {split_col}",
                fontsize=16,
                fontweight="bold",
            )
            plt.ylabel(f"Percentage of {target_value}", fontsize=14)
            plt.xlabel(split_col, fontsize=14)
            plt.ylim(0, max(value_percent[f"{target_value} Percentage (%)"]) * 1.2)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            safe_name = str(target_value).replace(" ", "_").replace("+", "plus")
            plt.savefig(
                os.path.join(
                    output_dir, f"{prefix}{safe_name}_percent_by_{split_col}.png"
                ),
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()

    # Summary statistics keyed by display names
    stats = {
        "total_items": len(filtered),
        f"{target_col}_counts": filtered[display_col].value_counts().to_dict(),
        f"{target_col}_percentages": (
            filtered[display_col].value_counts(normalize=True) * 100
        ).to_dict(),
    }
    if split_col:
        stats[f"{target_col}_by_{split_col}"] = {
            split_val: filtered[filtered[split_col] == split_val][display_col]
            .value_counts()
            .to_dict()
            for split_val in filtered[split_col].unique()
        }
        for target_value in target_values:
            stats[f"{target_value}_percentage_by_{split_col}"] = (
                filtered.groupby(split_col)[display_col]
                .apply(lambda x: sum(x == target_value) / len(x) * 100)
                .to_dict()
            )
    return stats


# Feature selection function
def enhance_feature_selection(
    features_df,
    target_series,
    selected_features=None,
    remove_low_variance=True,
    variance_threshold=0.01,
    remove_correlated=True,
    correlation_threshold=0.95,
    select_k_best=None,
    output_dir=None,
    prefix=None,
):
    """Enhance feature selection by removing low variance features, highly correlated features, and selecting top K features based on ANOVA F-value.

    Args:
        features_df: pd.DataFrame DataFrame containing feature columns.
        target_series: pd.Series Series containing target labels for classification.
        selected_features: list or None List of initial selected feature column names. If None, use all features.
        remove_low_variance: bool Whether to remove low variance features.
        variance_threshold: float Variance threshold below which features are removed.
        remove_correlated: bool Whether to remove highly correlated features.
        correlation_threshold: float Correlation threshold above which one of the features is removed.
        select_k_best: int or None Number of top features to select based on ANOVA F-value. If None, skip this step.
        output_dir: str or None Directory to save plots and feature scores. If None, no plots are saved.
        prefix: str or None Prefix for saved plot filenames. If None, no prefix is added.

    Returns:
        list List of final selected feature column names after enhancement.
    """
    # Use provided features or all features
    if selected_features is None:
        selected_features = features_df.columns.tolist()

    # Start with selected features
    X = features_df[selected_features]
    y = target_series

    # Final selected features
    final_features = selected_features.copy()
    original_count = len(final_features)

    # Step 1: Remove low variance features
    if remove_low_variance:
        # Apply variance threshold
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X)
        # Get mask of selected features
        support = selector.get_support()
        # Update feature list
        low_var_removed = [feat for i, feat in enumerate(final_features) if support[i]]

        print(
            f"Removed {len(final_features) - len(low_var_removed)} low variance features"
        )
        final_features = low_var_removed

    # Step 2: Remove highly correlated features
    if remove_correlated and len(final_features) > 1:
        # Calculate correlation matrix
        X_filtered = features_df[final_features]
        corr_matrix = X_filtered.corr().abs()

        # Create a mask for the upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )

        # Find features with correlation greater than threshold
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > correlation_threshold)
        ]

        # Update feature list
        corr_removed = [feat for feat in final_features if feat not in to_drop]

        print(
            f"Removed {len(final_features) - len(corr_removed)} highly correlated features"
        )
        final_features = corr_removed

        # Plot correlation matrix if output directory provided
        if output_dir and prefix:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(15, 15))
            sns.heatmap(
                features_df[final_features].corr(),
                annot=False,
                cmap="cell_classifier",
                vmin=-1,
                vmax=1,
                square=True,
            )
            plt.title(
                "Feature Correlation Matrix After Filtering",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{prefix}_correlation_matrix.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # Step 3: Select top K features based on ANOVA F-value
    if select_k_best and select_k_best < len(final_features):
        X_filtered = features_df[final_features]

        # Apply SelectKBest with f_classif (ANOVA F-value) for classification
        selector = SelectKBest(f_classif, k=select_k_best)
        selector.fit(X_filtered, y)

        # Get selected feature indices
        support = selector.get_support()

        # Get feature scores
        scores = selector.scores_

        # Extract selected features
        anova_selected = [feat for i, feat in enumerate(final_features) if support[i]]

        # Create a dataframe with feature names and scores
        feature_scores = pd.DataFrame({"Feature": final_features, "Score": scores})
        feature_scores = feature_scores.sort_values(by="Score", ascending=False)

        print(f"Selected top {len(anova_selected)} features based on ANOVA F-value")
        final_features = anova_selected

        # Plot feature importance if output directory provided
        if output_dir and prefix:
            num_features_to_plot = min(50, len(feature_scores))
            plt.figure(
                figsize=(12, max(8, num_features_to_plot / 3))
            )  # Adjust height based on feature count

            # Color mapping based on score percentile
            max_score = feature_scores["Score"].max()
            feature_scores["Normalized_Score"] = feature_scores["Score"] / max_score

            # Create a color gradient for the bars
            colors = [
                plt.cm.viridis(score)
                for score in feature_scores.head(num_features_to_plot)[
                    "Normalized_Score"
                ]
            ]

            # Plot with enhanced styling
            ax = sns.barplot(
                x="Score",
                y="Feature",
                data=feature_scores.head(num_features_to_plot),
                palette=colors,
            )

            plt.title("Top Features by ANOVA F-value", fontsize=16, fontweight="bold")
            plt.xlabel("F-value Score", fontsize=14)
            plt.ylabel("Feature", fontsize=14)

            # Add value labels
            for i, v in enumerate(feature_scores.head(num_features_to_plot)["Score"]):
                ax.text(v + max_score * 0.01, i, f"{v:.1f}", va="center", fontsize=9)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{prefix}_top_features.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Save feature scores
            feature_scores.to_csv(
                os.path.join(output_dir, f"{prefix}_feature_scores.csv"), index=False
            )

    print(
        f"Feature selection: {original_count} features -> {len(final_features)} features"
    )

    return final_features


def filter_classes(class_labels, class_mapping, REMOVE_MASK_LABELS):
    """Filter out specified class labels from class_labels and class_mapping.

    Args:
        class_labels: list List of current class labels (strings).
        class_mapping: dict Dictionary with 'label_to_class' mapping (int to str).
        REMOVE_MASK_LABELS: list or None List of class labels (strings) to remove.
            If None or empty, no filtering is applied.

    Returns:
        tuple: (new_class_labels, new_class_mapping, preserved_class_ids)
            - new_class_labels: list of class labels after removal.
            - new_class_mapping: dict with updated 'label_to_class' mapping.
            - preserved_class_ids: list of int class IDs that are preserved.
    """
    # 1) Validate inputs
    if "label_to_class" not in class_mapping or not isinstance(
        class_mapping["label_to_class"], dict
    ):
        raise ValueError("class_mapping must be {'label_to_class': {int: str, ...}}")

    label_to_class = class_mapping["label_to_class"]
    all_known_labels = set(class_labels) | set(label_to_class.values())

    # 2) No filtering case (None or empty)
    if not REMOVE_MASK_LABELS:  # handles None, [], (), set()
        preserved_class_ids = sorted(label_to_class.keys())
        print("=== No Filtering Requested ===")
        print("REMOVE_MASK_LABELS is None or empty. Preserving all classes.")
        print(f"Class labels: {class_labels}")
        print(f"Mapping (id -> label): {dict(sorted(label_to_class.items()))}")
        print(f"Preserved class IDs: {preserved_class_ids}")
        # Return copies to avoid accidental mutation of inputs
        return (
            list(class_labels),
            {"label_to_class": dict(label_to_class)},
            preserved_class_ids,
        )

    # 3) Validate labels to remove
    to_remove = list(REMOVE_MASK_LABELS)
    missing = [lbl for lbl in to_remove if lbl not in all_known_labels]
    if missing:
        raise ValueError(f"Labels to remove not found: {missing}")

    to_remove_set = set(to_remove)

    # 4) Remove from class_labels (order preserved)
    new_class_labels = [lbl for lbl in class_labels if lbl not in to_remove_set]

    # 5) Remove from class_mapping['label_to_class'] (keep original ids for retained labels)
    new_label_to_class = {
        k: v for k, v in label_to_class.items() if v not in to_remove_set
    }

    # 6) Preserved class IDs
    preserved_class_ids = sorted(new_label_to_class.keys())

    # 7) Diagnostics
    removed_ids = sorted([k for k, v in label_to_class.items() if v in to_remove_set])
    removed_pairs = {k: label_to_class[k] for k in removed_ids}
    retained_pairs = dict(sorted(new_label_to_class.items()))

    # 8) Print clear summaries
    print("=== Removal Summary ===")
    print(f"Requested to remove labels: {sorted(to_remove)}")
    print(f"Removed (id -> label): {removed_pairs}")
    print()
    print("=== Retained ===")
    print(f"Retained labels (order preserved): {new_class_labels}")
    print(f"Retained mapping (id -> label): {retained_pairs}")
    print(f"Class IDs: {preserved_class_ids}")

    # 9) Return updated structures and IDs
    return new_class_labels, {"label_to_class": new_label_to_class}, preserved_class_ids


class SciKitCellClassifier(CellClassifier):
    """Cell classifier using a SciKit-Learn compatible model.

    Args:
        model: Trained SciKit-Learn compatible model with predict and predict_proba methods.
        features: List of feature column names used for classification.
        target_col: Name of the target column to add to metadata (e.g., 'predicted_phase').
        label_encoder: Optional label encoder used during training (for models like xgb/lgb).
        class_id_to_name: Optional mapping from numeric class IDs to display names (strings).

    Returns:
        SciKitCellClassifier instance.
    """

    def __init__(
        self, model, features, target_col, label_encoder=None, class_id_to_name=None
    ):
        """Initialize the SciKitCellClassifier with a trained model and feature information.

        Args:
            model: Trained SciKit-Learn compatible model with predict and predict_proba methods.
            features: List of feature column names used for classification.
            target_col: Name of the target column to add to metadata (e.g., 'predicted_phase').
            label_encoder: Optional label encoder used during training (for models like xgb/lgb).
            class_id_to_name: Optional mapping from numeric class IDs to display names (strings).

        Returns:
            SciKitCellClassifier instance.
        """
        self.model = model
        self.features = features
        self.target_col = target_col
        self.label_encoder = label_encoder
        # Mapping from numeric class id -> display name (string)
        self.class_id_to_name = class_id_to_name or {}

        # Try to expose classes_ if available on the underlying estimator
        if hasattr(model, "classes_"):
            self.classes_ = model.classes_
        elif hasattr(model, "named_steps"):
            # Try last step
            last_step = list(model.named_steps.values())[-1]
            self.classes_ = getattr(last_step, "classes_", None)
        else:
            self.classes_ = None

    def classify_cells(self, metadata_df, features_df):
        """Classify cells based on feature data. Adds predicted class and confidence to metadata.

        Args:
            metadata_df: pd.DataFrame DataFrame containing cell metadata.
            features_df: pd.DataFrame DataFrame containing feature columns for cells.

        Returns:
            tuple: (metadata_df, features_df) with added classification results in metadata_df.
        """
        feature_cols = self.features
        missing_features = [f for f in feature_cols if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        X = features_df[feature_cols]
        valid_mask = ~X.isna().any(axis=1)
        if not valid_mask.all():
            removed_count = (~valid_mask).sum()
            print(f"Removing {removed_count} rows with NaN values in features")
            X = X.loc[valid_mask]
            metadata_df = metadata_df.loc[valid_mask]
            features_df = features_df.loc[valid_mask]
        if len(X) == 0:
            print("Warning: No valid rows found for classification")
            return metadata_df, features_df

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        # If label encoder used during training (e.g., xgb/lgb), inverse transform back to original labels
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)

        confidences = np.max(y_prob, axis=1)

        result = pd.DataFrame(index=X.index)
        result[self.target_col] = y_pred
        result[f"{self.target_col}_confidence"] = confidences

        metadata_df = pd.concat([metadata_df, result], axis=1)
        return metadata_df, features_df

    @classmethod
    def from_training(
        cls,
        metadata_df,
        features_df,
        target_column,
        selected_features=None,
        model_type="svc",
        scaler_type="standard",
        do_grid_search=True,
        class_mapping=None,
        output_dir=None,
        prefix=None,
        plot_results=True,
        retrain_on_full_data=True,
        enhance_features=False,
        remove_low_variance=True,
        variance_threshold=0.01,
        remove_correlated=True,
        correlation_threshold=0.95,
        select_k_best=None,
    ):
        """Create and train a classifier, saving artifacts and returning a SciKitCellClassifier."""
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            model_dir = os.path.join(output_dir, prefix) if prefix else output_dir
            os.makedirs(model_dir, exist_ok=True)
            feature_dir = os.path.join(model_dir, "features")
            os.makedirs(feature_dir, exist_ok=True)
            eval_dir = os.path.join(model_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
        else:
            model_dir = None
            feature_dir = None
            eval_dir = None

        # Check if target column exists in metadata
        if target_column not in metadata_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in metadata_df"
            )

        # Use provided features or all features
        if selected_features is None:
            selected_features = features_df.columns.tolist()

        # Check if features exist in the dataframe
        missing_features = [
            f for f in selected_features if f not in features_df.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Apply enhanced feature selection if requested
        if enhance_features:
            selected_features = enhance_feature_selection(
                features_df=features_df,
                target_series=metadata_df[target_column],
                selected_features=selected_features,
                remove_low_variance=remove_low_variance,
                variance_threshold=variance_threshold,
                remove_correlated=remove_correlated,
                correlation_threshold=correlation_threshold,
                select_k_best=select_k_best,
                output_dir=feature_dir,
                prefix=prefix,
            )
            if feature_dir:
                feature_path = os.path.join(model_dir, f"{prefix}_features.txt")
                with open(feature_path, "w") as f:
                    f.write("\n".join(selected_features))
                print(f"Saved selected features to {feature_path}")

        # Determine if we need label encoding (xgb/lgb typically need encoded 0..K-1 labels)
        needs_label_encoder = model_type in ["xgb", "lgb"]
        label_encoder = None
        encoded_classes = None
        if needs_label_encoder:
            label_encoder = LabelEncoder()
            y_original = metadata_df[target_column]
            y = label_encoder.fit_transform(y_original)
            encoded_classes = label_encoder.classes_
        else:
            y = metadata_df[target_column]
            y_original = y

        unique_classes = np.unique(y_original)
        is_binary = len(unique_classes) == 2

        # Scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "none":
            scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        pipeline_steps = []
        if scaler is not None:
            pipeline_steps.append((scaler_type + "_scaler", scaler))

        if model_type == "svc":
            param_grid = {
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", "auto", 0.1, 0.01],
                "svc__kernel": ["rbf", "linear"],
            }
            pipeline_steps.append(
                ("svc", SVC(probability=True, class_weight="balanced"))
            )
        elif model_type == "rf":
            param_grid = {
                "rf__n_estimators": [100, 200, 300],
                "rf__max_depth": [None, 10, 20, 30],
            }
            pipeline_steps.append(
                ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
            )
        elif model_type == "xgb":
            param_grid = {
                "xgb__n_estimators": [100, 200, 300],
                "xgb__max_depth": [3, 5, 7, 10],
                "xgb__learning_rate": [0.01, 0.1, 0.2],
            }
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_original), y=y_original
            )
            class_weight_dict = {
                cls: weight for cls, weight in zip(np.unique(y_original), class_weights)
            }
            if is_binary:
                neg_pos_ratio = len(y_original[y_original != unique_classes[1]]) / len(
                    y_original[y_original == unique_classes[1]]
                )
                pipeline_steps.append(
                    (
                        "xgb",
                        XGBClassifier(
                            random_state=42,
                            eval_metric="logloss",
                            objective="binary:logistic",
                            scale_pos_weight=neg_pos_ratio,
                        ),
                    )
                )
            else:
                pipeline_steps.append(
                    (
                        "xgb",
                        XGBClassifier(
                            random_state=42,
                            eval_metric="mlogloss",
                            objective="multi:softprob",
                            num_class=len(np.unique(y_original)),
                        ),
                    )
                )
        elif model_type == "lgb":
            param_grid = {
                "lgb__n_estimators": [100, 200, 300],
                "lgb__max_depth": [3, 5, 7, 10],
                "lgb__learning_rate": [0.01, 0.05, 0.1],
            }
            if is_binary:
                pipeline_steps.append(
                    (
                        "lgb",
                        LGBMClassifier(
                            random_state=42,
                            verbose=-1,
                            objective="binary",
                            class_weight="balanced",
                        ),
                    )
                )
            else:
                pipeline_steps.append(
                    (
                        "lgb",
                        LGBMClassifier(
                            random_state=42,
                            verbose=-1,
                            objective="multiclass",
                            num_class=len(np.unique(y_original)),
                            class_weight="balanced",
                        ),
                    )
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        pipeline = Pipeline(pipeline_steps)
        X = features_df[selected_features]
        # y already computed based on needs_label_encoder

        if len(X) != len(y):
            raise ValueError(
                f"Features ({len(X)} rows) and target ({len(y)} rows) must have the same length"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        start_time = time.time()
        if do_grid_search:
            print(
                f"Performing grid search for '{target_column}' classification with {model_type}..."
            )
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring="balanced_accuracy",
                verbose=1,
                n_jobs=10,
            )
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print("Best parameters:", best_params)
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            print(
                f"Training {model_type} model for '{target_column}' classification..."
            )
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        y_pred = best_pipeline.predict(X_test)
        class_report = sklearn_classification_report(y_test, y_pred, output_dict=True)
        print("\nTest Set Classification Report:")
        print(sklearn_classification_report(y_test, y_pred))

        # Build mapping/order for display
        class_id_to_name = None
        class_ids_order = None
        display_names_order = None
        if (
            class_mapping
            and isinstance(class_mapping, dict)
            and "label_to_class" in class_mapping
        ):
            class_id_to_name = dict(class_mapping["label_to_class"])
            class_ids_order = list(class_id_to_name.keys())
            display_names_order = [class_id_to_name[cid] for cid in class_ids_order]
        else:
            # Fallback to sorted unique of original labels
            class_ids_order = sorted(np.unique(y_original).tolist())
            class_id_to_name = {cid: str(cid) for cid in class_ids_order}
            display_names_order = [class_id_to_name[cid] for cid in class_ids_order]

        # Determine the order of y labels to pass to confusion_matrix based on how y_test is encoded
        if needs_label_encoder:
            # y_test are encoded indices 0..K-1; map from original id -> encoded idx
            enc_index_for_orig = {
                orig: idx for idx, orig in enumerate(label_encoder.classes_)
            }
            y_label_values_order = [
                enc_index_for_orig[cid]
                for cid in class_ids_order
                if cid in enc_index_for_orig
            ]
        else:
            # y_test are the original ids
            y_label_values_order = class_ids_order

        if plot_results and eval_dir is not None:
            cls._plot_evaluation_results(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=best_pipeline.predict_proba(X_test),
                y_label_values=y_label_values_order,
                class_names_display=display_names_order,
                output_dir=eval_dir,
                prefix=prefix if prefix is not None else model_type,
            )
            if model_type in ["rf", "lgb", "xgb"] and feature_dir is not None:
                cls._plot_feature_importance(
                    best_pipeline,
                    selected_features,
                    output_dir=feature_dir,
                    prefix=prefix if prefix is not None else model_type,
                )

        if retrain_on_full_data:
            print("\nRetraining model on full dataset with optimal parameters...")
            final_pipeline = clone(best_pipeline) if do_grid_search else best_pipeline
            full_start_time = time.time()
            if model_type == "xgb" and not is_binary:
                # Optional: sample weights, but we keep behavior consistent
                final_pipeline.fit(X, y)
            else:
                final_pipeline.fit(X, y)
            full_training_time = time.time() - full_start_time
            print(
                f"Full dataset training completed in {full_training_time:.2f} seconds"
            )
            classifier = cls(
                model=final_pipeline,
                features=selected_features,
                target_col=target_column,
                label_encoder=label_encoder if needs_label_encoder else None,
                class_id_to_name=class_id_to_name,
            )
        else:
            classifier = cls(
                model=best_pipeline,
                features=selected_features,
                target_col=target_column,
                label_encoder=label_encoder if needs_label_encoder else None,
                class_id_to_name=class_id_to_name,
            )

        # Save quick prediction preview
        if len(X_test) >= 5:
            test_metadata = metadata_df.loc[X_test.iloc[:5].index]
            metadata_result, _ = classifier.classify_cells(
                test_metadata, X_test.iloc[:5]
            )
            preview_cols = [
                classifier.target_col,
                f"{classifier.target_col}_confidence",
            ]
            print("\nTest prediction examples:")
            print(metadata_result[preview_cols].head())

        if model_dir is not None and prefix is not None:
            model_path = os.path.join(model_dir, f"{prefix}_model.dill")
            classifier.save(model_path)
            print(f"Model saved to: {model_path}")

            # Save feature list at model root (for notebook consumption)
            feature_path = os.path.join(model_dir, f"{prefix}_features.txt")
            with open(feature_path, "w") as f:
                f.write("\n".join(selected_features))

            report_path = os.path.join(
                model_dir, f"{prefix}_classification_report.json"
            )
            with open(report_path, "w") as f:
                json.dump(class_report, f, indent=4)

            config = {
                "model_type": model_type,
                "scaler_type": scaler_type,
                "feature_count": len(selected_features),
                "do_grid_search": do_grid_search,
                "retrain_on_full_data": retrain_on_full_data,
                "target_column": target_column,
                "training_time": training_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "selected_features": selected_features,
                "class_ids_order": class_ids_order,
                "class_names_order": display_names_order,
            }
            config_path = os.path.join(model_dir, f"{prefix}_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

        return classifier

    @staticmethod
    def _plot_evaluation_results(
        y_true,
        y_pred,
        y_prob,
        y_label_values,
        class_names_display,
        output_dir=".",
        prefix="model",
    ):
        """Plot confusion matrix and ROC curves using a fixed label order and display names."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Confusion Matrix with controlled label order
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=y_label_values)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"
        ax = sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            xticklabels=class_names_display,
            yticklabels=class_names_display,
            cmap="Blues",
            vmin=0,
            annot_kws={"size": 12},
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("True Label", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), weight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), weight="bold")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ROC Curves
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.viridis
        num_classes = len(class_names_display)
        if num_classes == 2:
            # Identify positive column index based on y_label_values (second class)
            pos_index = 1
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, pos_index])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                label=f"ROC curve (area = {roc_auc:.2f})",
                color=cmap(0.5),
                linewidth=3,
            )
        else:
            # Build binary matrix for true labels in the given order
            y_bin = np.zeros((len(y_true), num_classes))
            # Map actual label values in y_true to positional indices in y_label_values
            label_to_pos = {lbl: i for i, lbl in enumerate(y_label_values)}
            for i, val in enumerate(y_true):
                if val in label_to_pos:
                    y_bin[i, label_to_pos[val]] = 1
            for i in range(num_classes):
                if np.sum(y_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    color_pos = i / (num_classes - 1) if num_classes > 1 else 0.5
                    plt.plot(
                        fpr,
                        tpr,
                        label=f"ROC {class_names_display[i]} (area = {roc_auc:.2f})",
                        linewidth=2,
                        color=cmap(color_pos),
                    )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(
            "Receiver Operating Characteristic (ROC)", fontsize=16, fontweight="bold"
        )
        leg = plt.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_frame().set_edgecolor("lightgray")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_roc_curve.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _plot_feature_importance(
        model, feature_names, top_n=20, output_dir=".", prefix="model"
    ):
        """Plot feature importance for tree-based models.

        Parameters:
            model: Trained scikit-learn Pipeline or estimator with feature_importances_ attribute.
            feature_names: List of feature names corresponding to the model input.
            top_n: Number of top features to display. If None, display all.
            output_dir: Directory to save the plot.
            prefix: Prefix for the output file name.

        Returns:
            None
        """
        # Check if model has feature importances
        has_importance = False

        # Try to get the actual model from the pipeline
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    estimator = step
                    has_importance = True
                    break
        elif hasattr(model, "feature_importances_"):
            estimator = model
            has_importance = True

        if not has_importance:
            print("Model doesn't have feature importances. Skipping importance plot.")
            return

        # Get feature importances
        importances = estimator.feature_importances_

        # Sort feature importances
        indices = np.argsort(importances)[::-1]

        # Create DataFrame for easier handling
        importance_df = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in indices],
                "Importance": importances[indices],
            }
        )

        # Select top N features
        if top_n is not None and top_n < len(feature_names):
            importance_df = importance_df.head(top_n)

        # Normalize importances for color mapping
        max_importance = importance_df["Importance"].max()
        importance_df["Normalized"] = importance_df["Importance"] / max_importance

        # Create color gradient
        colors = [plt.cm.viridis(val) for val in importance_df["Normalized"]]

        # Plot feature importances with enhanced styling
        plt.figure(
            figsize=(12, max(8, len(importance_df) * 0.3))
        )  # Adjust height based on feature count

        # Plot horizontal bars
        ax = plt.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color=colors,
            alpha=0.8,
            edgecolor="gray",
            linewidth=0.5,
        )

        # Add importance value labels
        for i, v in enumerate(importance_df["Importance"]):
            plt.text(
                v + max_importance * 0.01,  # Slight offset from end of bar
                i,
                f"{v:.4f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Enhance visual elements
        plt.xlabel("Importance", fontsize=14)
        plt.ylabel("Feature", fontsize=14)
        plt.title("Feature Importance", fontsize=16, fontweight="bold")

        # Add grid lines
        plt.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_feature_importance.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # Also save a CSV with the full feature importance ranking
        importance_df.drop("Normalized", axis=1).to_csv(
            os.path.join(output_dir, f"{prefix}_feature_importance.csv"), index=False
        )

        plt.close()


# Consolidated training pipeline function (designed to be moved into lib)
from typing import List, Dict, Any, Tuple, Optional
import os, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import linregress


def train_classifier_pipeline(
    data: pd.DataFrame,
    class_title: str,
    class_id: List[int],
    class_labels: List[str],
    filtered_class_mapping: Dict[str, Any],
    metadata_cols: List[str],
    feature_markers: Dict[str, bool],
    exclude_markers: List[str],
    training_name: str,
    model_configs: List[Tuple],
    classifier_output_dir: Path,
    training_channels: List[str],
    remove_nan: bool = True,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    generate_accuracy_bar: bool = True,
    generate_f1_heatmap: bool = True,
    generate_feature_vs_accuracy: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train classifiers for a specified categorical target using a structured pipeline.

    Args:
        data: pd.DataFrame DataFrame containing cell metadata and features.
        class_title: str Name of the target column in metadata for classification.
        class_id: list List of class IDs (integers) to include in training.
        class_labels: list List of class labels (strings) corresponding to class_id.
        filtered_class_mapping: dict Dictionary with 'label_to_class' mapping (int to str).
        metadata_cols: list List of metadata column names to retain.
        feature_markers: dict Dictionary specifying which feature markers to include (key: marker name,
            value: bool indicating inclusion).
        exclude_markers: list List of feature markers (strings) to exclude.
        training_name: str Descriptive name for the training run (used in outputs).
        model_configs: list of tuples Each tuple specifies a model configuration:
            (name, model_type, scaler_type) or
            (name, model_type, scaler_type, feature_config_dict)
            where feature_config_dict can have keys:
                - enhance: bool (whether to apply enhanced feature selection)
                - remove_low_variance: bool
                - variance_threshold: float
                - remove_correlated: bool
                - correlation_threshold: float
                - select_k_best: int or None
        classifier_output_dir: Path Directory path to save outputs (models, plots, stats).
        training_channels: list List of channel names (strings) used in training (for logging).
        remove_nan: bool Whether to remove features with NaN values. Default is True.
        variance_threshold: float Threshold for variance-based feature removal. Default is 0.01.
        correlation_threshold: float Threshold for correlation-based feature removal. Default is 0.95.
        generate_accuracy_bar: bool Whether to generate accuracy bar plot across models. Default is True.
        generate_f1_heatmap: bool Whether to generate F1-score heatmap across models and classes. Default is True.
        generate_feature_vs_accuracy: bool Whether to plot number of features vs accuracy. Default is True.
        verbose: bool Whether to print progress messages. Default is True.

    Returns:
        dict Dictionary containing:
            - 'models': List of trained SciKitCellClassifier instances.
            - 'results': List of result dictionaries for each model configuration.
    """
    if verbose:
        print(f"[pipeline] Starting training pipeline: {training_name}")

    # 1. Style + directories
    _ = setup_publication_plot_style()
    dirs = create_run_directories(classifier_output_dir)
    statistics_dir = dirs["statistics"]
    models_dir = dirs["models"]
    plots_dir = dirs["plots"]
    results_dir = dirs["results"]
    if verbose:
        print(f"[pipeline] Run dir: {dirs['run']}")

    # 2. Class filter
    data_filt = data[data[class_title].isin(class_id)].copy()
    if data_filt.empty:
        raise ValueError("Filtered dataset empty after applying class_id subset.")

    # 3. Split
    metadata_df, features_df = split_cell_data(data_filt, metadata_cols=metadata_cols)
    if verbose:
        print(f"[pipeline] Metadata: {metadata_df.shape} Features: {features_df.shape}")

    # 4. Feature selection
    selected_features = select_features_from_split(
        features_df,
        feature_markers=feature_markers,
        exclude_markers=exclude_markers,
        exclude_cols=None,
        remove_nan=remove_nan,
        verbose=verbose,
    )
    if verbose:
        print(f"[pipeline] Selected {len(selected_features)} features")

    # 5. Distribution stats
    stats = plot_distribution_statistics(
        metadata_df,
        target_col=class_title,
        split_col="plate",
        output_dir=statistics_dir,
        prefix=training_name,
        target_order=class_labels,
        class_mapping=filtered_class_mapping,
        figsize=(12, 8),
        palette=None,
        dpi=300,
    )

    if verbose:
        print(
            f"\n=== Training '{class_title}' classifiers for '{training_name}' (channels: {', '.join(training_channels)}) ===\n"
        )

    # 6. Model training loop
    multiclass_results: List[Dict[str, Any]] = []
    ordered_id_name = list(filtered_class_mapping["label_to_class"].items())

    for config in model_configs:
        if len(config) == 3:
            name, model_type, scaler_type = config
            feature_config = None
        else:
            name, model_type, scaler_type, feature_config = config
        model_name = f"multiclass_{name}"
        if verbose:
            print(f"\n{'-' * 50}\nTraining model: {model_name}\n{'-' * 50}")

        # Base enhancement defaults
        enhance_params = {
            "enhance_features": False,
            "remove_low_variance": True,
            "variance_threshold": variance_threshold,
            "remove_correlated": True,
            "correlation_threshold": correlation_threshold,
            "select_k_best": None,
        }
        if feature_config:
            if feature_config.get("enhance"):
                enhance_params["enhance_features"] = True
            for k in ("remove_low_variance", "remove_correlated", "select_k_best"):
                if k in feature_config:
                    enhance_params[k] = feature_config[k]

        try:
            model = SciKitCellClassifier.from_training(
                metadata_df=metadata_df,
                features_df=features_df,
                target_column=class_title,
                selected_features=selected_features,
                model_type=model_type,
                scaler_type=scaler_type,
                do_grid_search=False,
                class_mapping=filtered_class_mapping,
                output_dir=models_dir,
                prefix=model_name,
                plot_results=True,
                retrain_on_full_data=True,
                enhance_features=enhance_params["enhance_features"],
                remove_low_variance=enhance_params["remove_low_variance"],
                variance_threshold=enhance_params["variance_threshold"],
                remove_correlated=enhance_params["remove_correlated"],
                correlation_threshold=enhance_params["correlation_threshold"],
                select_k_best=enhance_params["select_k_best"],
            )

            # Report path
            report_fp = os.path.join(
                models_dir, model_name, f"{model_name}_classification_report.json"
            )
            with open(report_fp, "r") as f:
                report_dict = json.load(f)

            feat_sel_desc = (
                "none"
                if not feature_config
                else (
                    ("var" if feature_config.get("remove_low_variance") else "")
                    + ("corr" if feature_config.get("remove_correlated") else "")
                    + (
                        f"k{feature_config.get('select_k_best')}"
                        if feature_config.get("select_k_best")
                        else ""
                    )
                )
                or "none"
            )

            metrics = {
                "model": model_name,
                "model_type": model_type,
                "scaler_type": scaler_type,
                "feature_selection": feat_sel_desc,
                "accuracy": report_dict.get("accuracy"),
                "macro_avg_precision": report_dict.get("macro avg", {}).get(
                    "precision"
                ),
                "macro_avg_recall": report_dict.get("macro avg", {}).get("recall"),
                "macro_avg_f1": report_dict.get("macro avg", {}).get("f1-score"),
            }

            # Handle encoded labels for certain model types
            if model_type in ["xgb", "lgb"]:
                le = getattr(model, "label_encoder", None)
                orig_to_idx = {}
                if le is not None and hasattr(le, "classes_"):
                    for idx, orig in enumerate(le.classes_):
                        key = int(orig) if isinstance(orig, (np.integer, int)) else orig
                        orig_to_idx[key] = idx
                for cid, cname in ordered_id_name:
                    key = None
                    if str(cid) in report_dict:
                        key = str(cid)
                    elif cid in orig_to_idx and str(orig_to_idx[cid]) in report_dict:
                        key = str(orig_to_idx[cid])
                    elif cname in report_dict:
                        key = cname
                    if key and key in report_dict:
                        metrics[f"{cname}_precision"] = report_dict[key].get(
                            "precision"
                        )
                        metrics[f"{cname}_recall"] = report_dict[key].get("recall")
                        metrics[f"{cname}_f1"] = report_dict[key].get("f1-score")
            else:
                for cid, cname in ordered_id_name:
                    key = None
                    if str(cid) in report_dict:
                        key = str(cid)
                    elif cname in report_dict:
                        key = cname
                    if key and key in report_dict:
                        metrics[f"{cname}_precision"] = report_dict[key].get(
                            "precision"
                        )
                        metrics[f"{cname}_recall"] = report_dict[key].get("recall")
                        metrics[f"{cname}_f1"] = report_dict[key].get("f1-score")

            feat_file = os.path.join(
                models_dir, model_name, f"{model_name}_features.txt"
            )
            if os.path.exists(feat_file):
                with open(feat_file, "r") as f:
                    used_features = f.read().splitlines()
                metrics["feature_count"] = len(used_features)
            else:
                metrics["feature_count"] = len(selected_features)

            multiclass_results.append(metrics)
            if verbose:
                print(f"[pipeline] Trained {model_name}")

        except Exception as e:
            print(f"[pipeline][ERROR] {model_name}: {e}")
            continue

    metrics_df = pd.DataFrame(multiclass_results)
    if not metrics_df.empty:
        metrics_out = os.path.join(results_dir, "multiclass_classifier_results.csv")
        metrics_df.to_csv(metrics_out, index=False)
        if verbose:
            print(f"[pipeline] Saved metrics CSV: {metrics_out}")

    # 7. Plots
    if verbose:
        print("\n=== Generating Result Comparisons ===\n")

    if not metrics_df.empty:
        # Accuracy bar
        if generate_accuracy_bar:
            plt.figure(figsize=(12, 8))
            model_types = metrics_df["model_type"].unique()
            color_positions = np.linspace(0.1, 0.9, len(model_types))
            model_type_colors = {
                m: plt.cm.viridis(pos) for m, pos in zip(model_types, color_positions)
            }
            ax = sns.barplot(
                x="model",
                y="accuracy",
                hue="model_type",
                data=metrics_df,
                palette=model_type_colors,
                alpha=0.8,
            )
            for p in ax.patches:
                h = p.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plt.title(
                f"{class_title}: Accuracy Comparison across Models",
                fontsize=16,
                fontweight="bold",
            )
            plt.ylabel("Accuracy")
            plt.xlabel("Model")
            leg = plt.legend(
                title="Model Type", frameon=True, fancybox=True, framealpha=0.9
            )
            leg.get_title().set_fontweight("bold")
            plt.tight_layout()
            out_fp = os.path.join(plots_dir, f"{training_name}_accuracy_comparison.png")
            plt.savefig(out_fp, dpi=300, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"[pipeline] Saved plot: {out_fp}")

        # F1 heatmap
        if generate_f1_heatmap:
            display_order = [
                v for _, v in filtered_class_mapping["label_to_class"].items()
            ]
            f1_cols = [
                f"{name}_f1"
                for name in display_order
                if f"{name}_f1" in metrics_df.columns
            ]
            if f1_cols:
                f1_data = metrics_df[["model"] + f1_cols].copy()
                f1_data.columns = ["model"] + [c.replace("_f1", "") for c in f1_cols]
                f1_pivot = f1_data.set_index("model")
                plt.figure(figsize=(14, 10))
                ax = sns.heatmap(
                    f1_pivot,
                    annot=True,
                    cmap="viridis",
                    fmt=".3f",
                    annot_kws={"fontsize": 10, "fontweight": "bold"},
                    linewidths=0.5,
                    linecolor="white",
                    vmin=0.0,
                    vmax=1.0,
                )
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label("F1 Score", fontsize=12, fontweight="bold")
                plt.title(
                    f"{class_title}: F1 Scores by Model and Class",
                    fontsize=16,
                    fontweight="bold",
                )
                plt.tight_layout()
                out_fp = os.path.join(plots_dir, f"{training_name}_f1_heatmap.png")
                plt.savefig(out_fp, dpi=300, bbox_inches="tight")
                plt.close()
                if verbose:
                    print(f"[pipeline] Saved plot: {out_fp}")

        # Feature count vs accuracy
        if generate_feature_vs_accuracy and "feature_count" in metrics_df.columns:
            plt.figure(figsize=(12, 8))
            model_types = metrics_df["model_type"].unique()
            markers = ["o", "s", "^", "d", "v"]
            for i, mtype in enumerate(model_types):
                subset = metrics_df[metrics_df["model_type"] == mtype]
                color_pos = i / (len(model_types) - 1) if len(model_types) > 1 else 0.5
                color = plt.cm.viridis(color_pos)
                marker = markers[i % len(markers)]
                plt.scatter(
                    subset["feature_count"],
                    subset["accuracy"],
                    label=mtype,
                    color=color,
                    marker=marker,
                    alpha=0.8,
                    s=150,
                    edgecolors="black",
                    linewidths=0.5,
                )
            for _, row in metrics_df.iterrows():
                plt.annotate(
                    row["model"].replace("multiclass_", ""),
                    (row["feature_count"], row["accuracy"]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )
            if len(metrics_df) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    metrics_df["feature_count"], metrics_df["accuracy"]
                )
                x_range = np.array(
                    [
                        metrics_df["feature_count"].min(),
                        metrics_df["feature_count"].max(),
                    ]
                )
                plt.plot(
                    x_range,
                    intercept + slope * x_range,
                    "k--",
                    alpha=0.6,
                    label=f"R² = {r_value**2:.3f}",
                )
            leg = plt.legend(
                title="Model Type",
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                loc="best",
            )
            leg.get_title().set_fontweight("bold")
            plt.xlabel("Number of Features")
            plt.ylabel("Accuracy")
            plt.title(
                f"{training_name}: Feature Count vs. Accuracy",
                fontsize=16,
                fontweight="bold",
            )
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            out_fp = os.path.join(
                plots_dir, f"{training_name}_feature_count_vs_accuracy.png"
            )
            plt.savefig(out_fp, dpi=300, bbox_inches="tight")
            plt.close()
            if verbose:
                print(f"[pipeline] Saved plot: {out_fp}")

    if verbose:
        print("[pipeline] Complete.")

    return {
        "metrics_df": metrics_df,
        "selected_features": selected_features,
        "stats": stats,
        "models_run": [m["model"] for m in multiclass_results],
        "dirs": dirs,
    }
