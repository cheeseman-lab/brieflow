"""Regenerate dummy_classifier.dill using the current Python environment.

Run with the brieflow_zarr environment:
    /lab/barcheese01/etopkoc/miniconda3/envs/brieflow_zarr/bin/python3 regen_dummy_classifier.py
"""
import sys
import os

# Add workflow lib to path so CellClassifier can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
workflow_lib = os.path.join(script_dir, "..", "..", "..", "workflow", "lib")
sys.path.insert(0, os.path.join(script_dir, "..", "..", "..", "workflow"))

import numpy as np
import dill
from lib.aggregate.cell_classification import CellClassifier


class DummyMitoticClassifier(CellClassifier):
    """Deterministic dummy classifier for testing.

    Assigns ~20% of cells to Mitotic (class_id=1) and ~80% to Interphase (class_id=2),
    using a fixed random seed for reproducibility.
    """

    def __init__(
        self,
        target_col="cell_stage",
        mitotic_fraction=0.2,
        seed=42,
        class_id_to_name=None,
        features=None,
    ):
        self.target_col = target_col
        self.mitotic_fraction = mitotic_fraction
        self.seed = seed
        self.class_id_to_name = class_id_to_name or {1: "Mitotic", 2: "Interphase"}
        self.features = features

    def classify_cells(self, metadata, features):
        rng = np.random.default_rng(self.seed)
        n = len(metadata)
        class_ids = np.where(rng.random(n) < self.mitotic_fraction, 1, 2).astype(int)
        confidence = rng.uniform(0.5, 1.0, n).astype(np.float32)
        confidence_col = f"{self.target_col}_confidence"
        metadata = metadata.copy()
        metadata[self.target_col] = class_ids
        metadata[confidence_col] = confidence
        return metadata, features


if __name__ == "__main__":
    clf = DummyMitoticClassifier()
    out_path = os.path.join(script_dir, "dummy_classifier.dill")
    clf.save(out_path)
    print(f"Saved to {out_path}")

    # Verify it loads back correctly
    loaded = CellClassifier.load(out_path)
    print(f"Loaded: {type(loaded)}")
    print(f"  target_col:       {loaded.target_col}")
    print(f"  mitotic_fraction: {loaded.mitotic_fraction}")
    print(f"  class_id_to_name: {loaded.class_id_to_name}")
    print("OK")
