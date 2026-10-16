"""calculate_ic_field samples at least one tile, so a well with few tiles still gets a field."""

import sys
from pathlib import Path

import numpy as np
import tifffile

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.illumination_correction import calculate_ic_field  # noqa: E402


def test_small_fraction_of_few_tiles_still_samples_one(tmp_path):
    files = []
    for i in range(3):
        fp = tmp_path / f"tile_{i}.tif"
        tifffile.imwrite(fp, np.full((1, 64, 64), 100 + i, dtype=np.uint16))
        files.append(str(fp))
    field = calculate_ic_field(files, sample_fraction=0.1, smooth=2)
    assert field.shape[-2:] == (64, 64)
    assert np.isfinite(field).all()
