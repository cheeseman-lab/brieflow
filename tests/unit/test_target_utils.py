from pathlib import Path

import pandas as pd

from workflow.lib.shared.target_utils import output_to_input
from workflow.lib.shared.file_utils import get_filename


# Verify output_to_input() supports metadata expansion of path-based file names
def test_output_to_input_supports_path_template_with_metadata_expansion():
    """
    Regression test for InputFunctionException where output_to_input() returned None
    when given a single Path template (not a list).
    """
    sbs_wildcard_combos = pd.DataFrame([{"plate": "1", "well": "A1", "tile": "2"}])

    sbs_fp = Path("brieflow_output") / "sbs"
    segmentation_stats_template = (
        sbs_fp
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "segmentation_stats",
            "tsv",
        )
    )

    res = output_to_input(
        segmentation_stats_template,
        wildcards={"plate": "1"},
        expansion_values=["well", "tile"],
        metadata_combos=sbs_wildcard_combos,
    )

    assert res == ["brieflow_output/sbs/tsvs/P-1_W-A1_T-2__segmentation_stats.tsv"]


def test_output_to_input_supports_list_of_one_template_with_metadata_expansion():
    sbs_wildcard_combos = pd.DataFrame([{"plate": "1", "well": "A1", "tile": "2"}])

    sbs_fp = Path("brieflow_output") / "sbs"
    combine_cells_template_list = [
        sbs_fp
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "cells", "parquet")
    ]

    res = output_to_input(
        combine_cells_template_list,
        wildcards={"plate": "1"},
        expansion_values=["well"],
        metadata_combos=sbs_wildcard_combos,
    )

    assert res == ["brieflow_output/sbs/parquets/P-1_W-A1__cells.parquet"]
