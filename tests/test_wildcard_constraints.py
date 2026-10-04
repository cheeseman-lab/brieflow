"""The {well} wildcard must not swallow a tile suffix.

TIFF basenames are `_`-delimited (`P-1_W-A1_T-0__reads.parquet`), so an
unconstrained `{well}` also matches `A1_T-0` and a per-well rule claims a
per-tile output that shares its directory — Snakemake raises
AmbiguousRuleException at DAG build, which no unit test reaches. Zarr mode
nests the location as directories, so it never sees the collision.
"""

import re
from pathlib import Path

import pytest

SNAKEFILE = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"


@pytest.mark.unit
def test_well_wildcard_cannot_swallow_a_tile_suffix():
    """Once a directory holds both per-tile and per-well tables, only the {well}
    constraint keeps their patterns apart."""
    match = re.search(r"^\s*well=r?\"([^\"]+)\",", SNAKEFILE.read_text(), re.M)
    assert match, "no {well} wildcard_constraints entry in workflow/Snakefile"
    well = re.compile(f"^{match.group(1)}$")
    assert well.match("A1"), "constraint must still accept a plain well"
    assert well.match("r02c02"), "constraint must still accept Opera Phenix wells"
    assert not well.match("A1_T-0"), "constraint must reject a tile-suffixed well"
