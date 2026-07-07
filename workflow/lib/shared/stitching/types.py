"""Data containers for the stitching core."""

from __future__ import annotations

import pandas as pd

_COLUMNS = ["tile", "y", "x"]


class TileOffsets:
    """Per-tile global-frame pixel offsets for one modality's well."""

    def __init__(self, frame: pd.DataFrame):
        """Store a copy of ``frame`` restricted to the offset columns."""
        self._frame = frame[_COLUMNS].reset_index(drop=True)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "TileOffsets":
        """Build from a DataFrame with columns tile, y, x."""
        return cls(frame)

    def to_frame(self) -> pd.DataFrame:
        """Return the offsets as a DataFrame with columns tile, y, x."""
        return self._frame.copy()
