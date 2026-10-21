"""Regression tests for `combinatorial.ambiguous_call: N` and the call_cells no-call options.

With the C = 488 + 568, A = 488 + 647 code, a cycle with all three dyes over the
fraction gate is one dye from both C and A, and the nearest-template call took
the first in code order, C. Measured on ten two-colour screen plates: 33-94% of C
calls were such cycles, their true base (leave-one-out against the library)
followed the library's base mix, and image crops showed nuclear haze rather
than spots. `ambiguous_call: N` calls them N instead; `n_wildcard` lets a read
with one N map when exactly one library barcode agrees at its other cycles, and
`recomb_max_n` compares the recombination barcode at its called cycles only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.sbs.call_cells import call_cells, resolve_wildcards  # noqa: E402
from lib.sbs.call_reads import call_reads  # noqa: E402

CODE = {
    "G": [],
    "T": ["Alexa 568"],
    "C": ["Alexa 488", "Alexa 568"],
    "A": ["Alexa 488", "Alexa 647"],
}
CHANNELS = ["Alexa 488", "Alexa 568", "Alexa 647"]
# X: all three dyes about equal, one dye from both C and A
LEVELS = {
    "G": (10, 10, 10),
    "T": (60, 900, 60),
    "C": (600, 900, 60),
    "A": (600, 60, 900),
    "X": (500, 500, 500),
}


def bases_frame(sequences):
    rows = []
    for read, seq in enumerate(sequences):
        for cycle, base in enumerate(seq, start=1):
            for channel, value in zip(CHANNELS, LEVELS[base]):
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


def reads(sequences, **combinatorial):
    out = call_reads(
        bases_data=bases_frame(sequences),
        method="frac",
        chemistry="combinatorial",
        combinatorial={"code": CODE, **combinatorial},
    )
    return out.sort_values("read").reset_index(drop=True)


def test_all_dyes_on_is_called_c_by_default():
    assert list(reads(["TCXA"]).barcode) == ["TCCA"]


def test_ambiguous_call_n_marks_the_tie():
    r = reads(["TCXA"], ambiguous_call="N")
    assert list(r.barcode) == ["TCNA"]
    assert r.loc[0, "Q_2"] == 0
    assert r.loc[0, "Q_min"] == r.loc[0, ["Q_0", "Q_1", "Q_3"]].min() > 0


def test_resolve_wildcards_only_when_unique():
    lib = pd.Series(["ACGT", "ACTT", "GGGG"])
    out = resolve_wildcards(pd.Series(["NGGG", "ACNT", "ACGT", "NNGG"]), lib, 1)
    assert list(out) == ["GGGG", "ACNT", "ACGT", "NNGG"]


def _cells(barcode, recomb_max_n, n_wildcard=1):
    reads_df = pd.DataFrame(
        dict(
            well=["r02c02"],
            tile=[1],
            cell=[1],
            read=[0],
            barcode=[barcode],
            peak=[100.0],
            **{f"Q_{i}": [0.0 if c == "N" else 1.0] for i, c in enumerate(barcode)},
        )
    )
    reads_df["Q_min"] = 1.0
    lib = pd.DataFrame(
        dict(
            prefix_map=["ACGT", "GGGG"],
            prefix_recomb=["TTAA", "CCAA"],
            gene_symbol=["X", "Y"],
            gene_id=[1, 2],
        )
    )
    return call_cells(
        reads_df,
        df_barcode_library=lib,
        map_start=1,
        map_end=4,
        recomb_start=5,
        recomb_end=8,
        recomb_filter_col="Q_recomb",
        recomb_q_thresh=0.1,
        n_wildcard=n_wildcard,
        recomb_max_n=recomb_max_n,
    )


def test_n_in_the_mapping_half_is_resolved():
    assert _cells("NCGTTTAA", 0).cell_barcode_0.iloc[0] == "ACGT"


def test_n_in_the_recombination_half_is_unknown_not_a_mismatch():
    assert bool(_cells("ACGTTNAA", 1).no_recomb_0.iloc[0]) is True
    assert bool(_cells("ACGTCNAA", 1).no_recomb_0.iloc[0]) is False
    assert pd.isna(_cells("ACGTTNAA", 0).no_recomb_0.iloc[0])


def test_all_n_read_has_zero_q_min():
    r = reads(["XXXX"], ambiguous_call="N")
    assert r.loc[0, "barcode"] == "NNNN"
    assert r.loc[0, "Q_min"] == 0
