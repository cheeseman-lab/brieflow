from abc import ABC, abstractmethod

import dill
from sklearn.preprocessing import RobustScaler
import numpy as np


class CellClassifier(ABC):
    """Base class for cell classifiers."""

    @abstractmethod
    def classify_cells(self, metadata, features):
        """Classify cells based on feature data.

        Takes DataFrames with metadata and features for cells.
        Uses features to determine the class of each cell and the confidence of that classification.
        The class and confidence are added to the metadata DataFrame.
        """
        pass

    def save(self, filename):
        """Save the classifier to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a classifier from a file."""
        with open(filename, "rb") as f:
            return dill.load(f)


class NaiveMitoticClassifier(CellClassifier):
    def __init__(self, threshold_variable="nucleus_DAPI_median", mitotic_threshold=4.5):
        self.threshold_variable = threshold_variable
        self.mitotic_threshold = mitotic_threshold

    def classify_cells(self, metadata, features):
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
        metadata = metadata.copy()
        metadata.loc[:, "class"] = classes
        metadata.loc[:, "confidence"] = confidences

        return metadata
