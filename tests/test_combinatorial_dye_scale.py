"""Regression tests for `combinatorial.dye_scale` in the frac base caller.

The frac caller gates each dye on at a single global fraction of the cycle's total
intensity. When one dye images dimmer than the others, its fraction in a two-dye base
state falls under that threshold on every cycle, so the two-dye base is called as the
one-dye base with no error raised. Measured on a PoTC RPE1 plate: 56% of single-base
errors were C called as T, with the Alexa 488 fraction at 0.15-0.17 against a 0.18
threshold, and rescaling 488 by 1.5 raised exact-library reads from 0.23 to 0.38.
`dye_scale` rescales a dye before fractions are formed; the caller stays library-blind.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.sbs.call_reads import call_reads  # noqa: E402

CODE = {
    "G": [],
    "T": ["Alexa 568"],
    "C": ["Alexa 488", "Alexa 568"],
    "A": ["Alexa 488", "Alexa 647"],
}
CHANNELS = ["Alexa 488", "Alexa 568", "Alexa 647"]
# Per base: dye intensities in CHANNELS order, on a plate where 488 images dim. In the C state
# (488 + 568) its fraction is 0.15, under the 0.18 gate, so C is called T; in the A state
# (488 + 647) the 647 partner is dimmer, so 488 clears the gate and A survives. G is dark.
DIM_488 = {
    "G": (10, 10, 10),
    "T": (60, 900, 60),
    "C": (170, 900, 60),
    "A": (170, 60, 400),
}


def bases_frame(sequences, intensities=DIM_488):
    rows = []
    for read, seq in enumerate(sequences):
        for cycle, base in enumerate(seq, start=1):
            for channel, value in zip(CHANNELS, intensities[base]):
                rows.append(
                    dict(
                        well="r02c02",
                        tile=1,
                        cell=read + 1,
                        read=read,
                        cycle=cycle,
                        channel=channel,
                        intensity=float(value),
                        i=read,
                        j=read,
                    )
                )
    return pd.DataFrame(rows)


def called(df, combinatorial):
    reads = call_reads(
        bases_data=df,
        method="frac",
        chemistry="combinatorial",
        combinatorial=combinatorial,
    )
    return list(reads.sort_values("read").barcode)


def test_dim_dye_is_miscalled_without_a_scale():
    """C (488 + 568) reads as T (568) when 488 is dim: the defect the option exists for."""
    assert called(bases_frame(["GTCA"]), {"code": CODE}) == ["GTTA"]


def test_dye_scale_recovers_the_dim_dye():
    combinatorial = {"code": CODE, "dye_scale": {"Alexa 488": 3.0}}
    assert called(bases_frame(["GTCA", "CTGA"]), combinatorial) == ["GTCA", "CTGA"]


def test_absent_or_unit_scale_is_a_no_op():
    df = bases_frame(["GTCA", "CTGA"])
    stock = called(df, {"code": CODE})
    assert called(df, {"code": CODE, "dye_scale": {}}) == stock
    assert (
        called(df, {"code": CODE, "dye_scale": {"Alexa 488": 1.0, "Alexa 647": 1.0}})
        == stock
    )


def test_scale_applies_per_dye_not_globally():
    """Scaling every dye equally cannot change a fraction, so the calls must not move."""
    df = bases_frame(["GTCA", "CTGA"])
    scaled = {label: 4.0 for label in CHANNELS}
    assert called(df, {"code": CODE, "dye_scale": scaled}) == called(df, {"code": CODE})


@pytest.mark.parametrize(
    "dye_scale, message",
    [
        ({"Alexa 999": 1.5}, "not in the combinatorial code"),
        ({"Alexa 488": 0}, "must be positive"),
        ({"Alexa 488": -1.0}, "must be positive"),
        ([("Alexa 488", 1.5)], "must be a mapping"),
    ],
)
def test_invalid_dye_scale_raises(dye_scale, message):
    with pytest.raises(ValueError, match=message):
        called(bases_frame(["GTCA"]), {"code": CODE, "dye_scale": dye_scale})


def test_blank_state_survives_the_scale():
    """A dark cycle stays G: the scale must not promote it to a called dye state."""
    calls = called(
        bases_frame(["GTCA", "GGTC"]), {"code": CODE, "dye_scale": {"Alexa 488": 3.0}}
    )
    assert [c[0] for c in calls] == ["G", "G"] and calls[1][1] == "G"


def test_an_oversized_scale_overcalls_the_dim_dye():
    """The scale is a calibration, not a free win: too large a factor calls T as C and
    starves 488's partner dye, so A loses its 647 and falls back to the blank state."""
    assert called(
        bases_frame(["GTCA"]), {"code": CODE, "dye_scale": {"Alexa 488": 12.0}}
    ) == ["GCCG"]


# --- brightness_regions -------------------------------------------------------
# A barcode whose two regions are imaged at different brightness: the blank gate compares
# each cycle to the median over ALL cycles, so cycles in the dim region read as blank. On a
# PoTC HeLa plate, taking the reference per region raised exact-library reads from 0.44 to
# 0.62 and mapped cells by 21%, with fewer decoy calls.
BRIGHT = {
    "G": (10, 10, 10),
    "T": (60, 900, 60),
    "C": (510, 900, 60),
    "A": (510, 60, 400),
}
DIM = {base: tuple(v / 6 for v in value) for base, value in BRIGHT.items()}


def two_region_frame(bright_seq, dim_seq):
    """One read: `bright_seq` on cycles 1..n then `dim_seq` on the cycles after it."""
    rows = []
    for cycle, base in enumerate(bright_seq + dim_seq, start=1):
        table = BRIGHT if cycle <= len(bright_seq) else DIM
        for channel, value in zip(CHANNELS, table[base]):
            rows.append(
                dict(
                    well="r02c02",
                    tile=1,
                    cell=1,
                    read=0,
                    cycle=cycle,
                    channel=channel,
                    intensity=float(value),
                    i=0,
                    j=0,
                )
            )
    return pd.DataFrame(rows)


def test_dim_region_reads_as_blank_without_regions():
    """The dim region's real bases are called G against a reference set by the bright region."""
    df = two_region_frame(["C", "T", "A", "C"], ["C", "T", "A"])
    assert called(df, {"code": CODE}) == ["CTACGGG"]


def test_brightness_regions_recovers_the_dim_region():
    combinatorial = {"code": CODE, "brightness_regions": [[1, 4], [5, 7]]}
    assert called(
        two_region_frame(["C", "T", "A", "C"], ["C", "T", "A"]), combinatorial
    ) == ["CTACCTA"]


def test_brightness_regions_absent_or_covering_everything_is_a_no_op():
    df = two_region_frame(["C", "T", "A", "C"], ["C", "T", "A"])
    stock = called(df, {"code": CODE})
    assert called(df, {"code": CODE, "brightness_regions": []}) == stock
    assert called(df, {"code": CODE, "brightness_regions": [[1, 7]]}) == stock


def test_cycles_outside_every_region_keep_the_global_reference():
    df = two_region_frame(["C", "T", "A", "C"], ["C", "T", "A"])
    partial = called(df, {"code": CODE, "brightness_regions": [[1, 4]]})
    assert partial == called(df, {"code": CODE})


def test_a_genuinely_dark_cycle_stays_blank_within_its_region():
    """Rescoping the reference must not turn a dark cycle into a called base."""
    df = two_region_frame(["C", "T", "A", "C"], ["G", "T", "A"])
    assert called(df, {"code": CODE, "brightness_regions": [[1, 4], [5, 7]]}) == [
        "CTACGTA"
    ]


@pytest.mark.parametrize(
    "regions, message",
    [
        ([[1]], "must be \\[start, end\\] cycle pairs"),
        ([[0, 4]], "1-based inclusive range"),
        ([[5, 2]], "1-based inclusive range"),
    ],
)
def test_invalid_brightness_regions_raise(regions, message):
    with pytest.raises(ValueError, match=message):
        called(bases_frame(["GTCA"]), {"code": CODE, "brightness_regions": regions})
