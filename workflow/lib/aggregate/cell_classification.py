import dill
from abc import ABC, abstractmethod


class CellClassifier(ABC):
    """Base class for cell classifiers."""

    @abstractmethod
    def classify_cells(self, cell_data, feature_start_idx):
        """Classify cells based on input data."""
        pass

    def save(self, filename):
        """Save the classifier to a file."""
        with open(filename, "wb") as f:
            dill.dump(self, f)  # Use dill instead of pickle

    @staticmethod
    def load(filename):
        """Load a classifier from a file."""
        with open(filename, "rb") as f:
            return dill.load(f)  # Use dill instead of pickle
