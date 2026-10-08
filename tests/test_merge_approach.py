"""The merge approach is validated when targets are built: `stitch` is gone, typos fail."""

import sys
from pathlib import Path

import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.target_utils import get_merge_targets_by_approach  # noqa: E402


@pytest.mark.parametrize(
    "approach,first",
    [
        (None, "fast_alignment"),
        ("fast", "fast_alignment"),
        ("positions", "positions_merge"),
    ],
)
def test_supported_approaches(approach, first):
    targets = get_merge_targets_by_approach({"merge": {"approach": approach}})
    assert targets[0] == first and "eval_merge" in targets


def test_stitch_points_to_positions():
    with pytest.raises(ValueError, match="positions"):
        get_merge_targets_by_approach({"merge": {"approach": "stitch"}})


def test_unknown_approach_fails():
    with pytest.raises(ValueError, match="fast' or 'positions"):
        get_merge_targets_by_approach({"merge": {"approach": "postions"}})
