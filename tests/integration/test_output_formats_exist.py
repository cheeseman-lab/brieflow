"""
Integration test ensuring both TIFF and Zarr outputs exist when configured.
"""
from pathlib import Path

from lib.shared.file_utils import get_filename


TEST_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "small_test_analysis"
ROOT_FP = TEST_ANALYSIS_PATH / "brieflow_output" / "preprocess"

TEST_PLATE = 1
TEST_WELL = "A1"
TEST_CYCLE = 11
TEST_TILE_SBS = 0


def test_both_tiff_and_zarr_sbs_outputs_exist():
    tiff_path = ROOT_FP / "images" / "sbs" / get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "tile": TEST_TILE_SBS,
            "cycle": TEST_CYCLE,
        },
        "image",
        "tiff",
    )
    zarr_path = ROOT_FP / "omezarr" / "sbs" / get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "tile": TEST_TILE_SBS,
            "cycle": TEST_CYCLE,
        },
        "image",
        "zarr",
    )
    assert tiff_path.exists(), f"Missing TIFF at {tiff_path}"
    assert zarr_path.exists() and zarr_path.is_dir(), f"Missing Zarr at {zarr_path}"


