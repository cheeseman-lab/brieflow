"""Cell classification module for the Brieflow workflow.

This module provides functionality to classify cells based on their features,
such as identifying mitotic cells vs interphase cells. It contains a base
classifier class and implementations of specific classifiers.
"""

from abc import ABC, abstractmethod

import dill
from sklearn.preprocessing import RobustScaler
import numpy as np

try:
    import fsspec
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False


class CellClassifier(ABC):
    """Base class for cell classifiers."""

    @abstractmethod
    def classify_cells(self, metadata, features):
        """Classify cells based on feature data.

        Takes DataFrames with metadata and features for cells.
        Uses features to determine the class of each cell and the confidence
        of that classification.

        Args:
            metadata (pd.DataFrame): DataFrame containing metadata for cells.
            features (pd.DataFrame): DataFrame containing feature data for cells.

        Returns:
            tuple: (metadata, features) - Modified DataFrames with classification
                  results added to metadata.
        """
        print("No classification method defined! Returning orginal cell data...")

    def save(self, filename):
        """Save the classifier to a file.

        Args:
            filename (str): Path to save the serialized classifier (supports GCS paths).
        """
        # Use fsspec for GCS paths or when available, otherwise use built-in open
        if HAS_FSSPEC and (filename.startswith("gs://") or filename.startswith("s3://")):
            with fsspec.open(filename, "wb") as f:
                dill.dump(self, f)
        else:
            with open(filename, "wb") as f:
                dill.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a classifier from a file.

        Args:
            filename (str): Path to the serialized classifier file (supports GCS paths).

        Returns:
            CellClassifier: The loaded classifier instance.
        """
        # Use fsspec for GCS paths or when available, otherwise use built-in open
        if HAS_FSSPEC and (filename.startswith("gs://") or filename.startswith("s3://")):
            with fsspec.open(filename, "rb") as f:
                return dill.load(f)
        else:
            with open(filename, "rb") as f:
                return dill.load(f)


class NaiveMitoticClassifier(CellClassifier):
    """Simple mitotic cell classifier based on threshold values.

    Classifies cells as either 'mitotic' or 'interphase' based on a threshold
    applied to a specified feature after robust scaling.
    """

    def __init__(self, threshold_variable="nucleus_DAPI_median", mitotic_threshold=4.5):
        """Initialize the naive mitotic classifier.

        Args:
            threshold_variable (str): The feature name to use for classification.
                Defaults to "nucleus_DAPI_median".
            mitotic_threshold (float): The threshold value above which cells are
                considered mitotic after robust scaling. Defaults to 4.5.
        """
        self.threshold_variable = threshold_variable
        self.mitotic_threshold = mitotic_threshold

    def classify_cells(self, metadata, features):
        """Classify cells as mitotic or interphase based on threshold.

        Args:
            metadata (pd.DataFrame): DataFrame containing metadata for cells.
            features (pd.DataFrame): DataFrame containing feature data for cells.

        Returns:
            tuple: (metadata, features) - Modified DataFrames with 'class' and
                  'confidence' columns added to metadata.
        """
        # Filter out rows with NA in threshold variable
        valid_mask = features[self.threshold_variable].notna()
        metadata = metadata.loc[valid_mask]
        features = features.loc[valid_mask]

        # Extract just the threshold variable we need
        threshold_values = features[self.threshold_variable].astype(float)

        # Scale only this single feature
        threshold_values_scaled = (
            RobustScaler()
            .fit_transform(threshold_values.values.reshape(-1, 1))
            .flatten()
        )

        # Classify cells
        classes = np.where(
            threshold_values_scaled > self.mitotic_threshold, "mitotic", "interphase"
        )

        # Compute confidence
        confidences = np.zeros(len(features))
        for cl in ["mitotic", "interphase"]:
            idx = np.where(classes == cl)[0]
            if len(idx) > 0:
                sorted_idx = idx[np.argsort(threshold_values.iloc[idx])]
                if cl == "interphase":
                    sorted_idx = sorted_idx[::-1]  # Reverse for interphase
                for rank, i in enumerate(sorted_idx):
                    confidences[i] = (rank + 1) / len(sorted_idx)

        # Insert class and confidence columns
        metadata.loc[:, "class"] = classes
        metadata.loc[:, "confidence"] = confidences

        return metadata, features
