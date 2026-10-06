"""The tsvs/ and parquets/ output directories must hold what they are named after.

PR #298 migrates per-tile SBS and phenotype tables from TSV to parquet by changing
the extension in `workflow/targets/*.smk`. The directory anchor sits on the same
line and is easy to leave behind, which silently produces `sbs/tsvs/*.parquet`.
The anchors are also load-bearing beyond tidiness: `visualization/src/filesystem.py`
keys its location chain off `LOCATION_ANCHORS = ("eval", "tsvs", "parquets")`.

Static check on the .smk text — the targets files reference Snakemake globals
(ROOT_FP, IMG_FMT, temp, the wildcard combos) and cannot be imported standalone.
"""

import re
from pathlib import Path

import pytest

TARGETS_DIR = Path(__file__).resolve().parents[1] / "workflow" / "targets"

ANCHOR_EXTENSION = {"tsvs": "tsv", "parquets": "parquet"}

# `FP / "<anchor>" / get_filename(` or `.../ get_data_output_path(`, whitespace-normalized.
_ANCHORED_CALL = re.compile(
    r'/\s*"(tsvs|parquets)"\s*/\s*(get_filename|get_data_output_path)\s*\('
)


def _call_args(text, open_paren_idx):
    """Return the argument text of the call whose '(' is at open_paren_idx."""
    depth = 0
    for i in range(open_paren_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : i]
    raise AssertionError(f"unbalanced parentheses at offset {open_paren_idx}")


def _anchored_outputs(path):
    """Yield (line_no, anchor, extension) for every anchored output in a .smk file."""
    text = path.read_text()
    for match in _ANCHORED_CALL.finditer(text):
        args = _call_args(text, match.end() - 1)
        # The extension is the last string literal in the call; the location dict
        # that precedes it may itself contain quoted wildcards.
        literals = re.findall(r'"([^"]*)"', args)
        assert literals, f"{path.name}: no string literal in {args!r}"
        yield text[: match.start()].count("\n") + 1, match.group(1), literals[-1]


@pytest.mark.unit
@pytest.mark.parametrize("smk", sorted(TARGETS_DIR.glob("*.smk")), ids=lambda p: p.name)
def test_output_directory_matches_file_extension(smk):
    mismatches = [
        f"{smk.name}:{line} writes .{ext} into {anchor}/ (expected .{ANCHOR_EXTENSION[anchor]})"
        for line, anchor, ext in _anchored_outputs(smk)
        if ext != ANCHOR_EXTENSION[anchor]
    ]
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.unit
def test_the_check_sees_the_sbs_tile_tables():
    """Guard the regex itself: if it stops matching, the test above passes vacuously."""
    found = {
        (anchor, ext) for _, anchor, ext in _anchored_outputs(TARGETS_DIR / "sbs.smk")
    }
    assert ("parquets", "parquet") in found
    assert ("tsvs", "tsv") in found
