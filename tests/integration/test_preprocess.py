"""
Test analysis for the preprocess module.
"""

from pathlib import Path

import pytest
import yaml
import pandas as pd
from tifffile import imread

from workflow.lib.shared.file_utils import get_filename
from workflow.lib.shared.io import read_image

TEST_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "small_test_analysis"
TEST_PLATE = 1
TEST_WELL = "A1"
TEST_CYCLE = 11
TEST_TILE_SBS = 0
TEST_TILE_PHENOTYPE = 5

# load config file
CONFIG_FILE_PATH = TEST_ANALYSIS_PATH / "config/config.yml"
with open(CONFIG_FILE_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


def _resolve_brieflow_output_dir() -> Path:
    """
    Integration tests read artifacts produced by the small-test Snakemake run.

    Prefer the canonical `tests/small_test_analysis/brieflow_output/`, but fall back to
    the newest `brieflow_output_*` directory if the canonical directory was deleted or
    not generated yet.
    """
    canonical = TEST_ANALYSIS_PATH / "brieflow_output"
    if canonical.exists():
        return canonical

    candidates = sorted(
        [
            p
            for p in TEST_ANALYSIS_PATH.iterdir()
            if p.is_dir() and p.name.startswith("brieflow_output_")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if (p / "preprocess" / "metadata").exists():
            return p

    pytest.skip(
        "Brieflow output directory not found. Run tests/small_test_analysis/run_brieflow.sh "
        "or tests/small_test_analysis/run_brieflow_omezarr.sh first."
    )


ROOT_FP = _resolve_brieflow_output_dir()
PREPROCESS_FP = ROOT_FP / "preprocess"


def test_combine_metadata_sbs():
    sbs_metadata_path = str(
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
            },
            "combined_metadata",
            "parquet",
        ),
    )
    sbs_metadata = pd.read_parquet(sbs_metadata_path)
    # Schema evolves; assert required columns rather than an exact count.
    required = {
        "x_pos",
        "y_pos",
        "z_pos",
        "pfs_offset",
        "plate",
        "well",
        "tile",
        "filename",
        "channels",
        "cycle",
    }
    assert required.issubset(set(sbs_metadata.columns))


def test_combine_metadata_phenotype():
    phenotype_metadata_path = str(
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
            },
            "combined_metadata",
            "parquet",
        ),
    )
    phenotype_metadata = pd.read_parquet(phenotype_metadata_path)
    required = {
        "x_pos",
        "y_pos",
        "z_pos",
        "pfs_offset",
        "plate",
        "well",
        "tile",
        "filename",
        "channels",
    }
    assert required.issubset(set(phenotype_metadata.columns))


def test_convert_sbs():
    # Try Zarr first (new default), then TIFF (legacy)
    base_path = PREPROCESS_FP / "images" / "sbs"
    filename_base = get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "tile": TEST_TILE_SBS,
            "cycle": TEST_CYCLE,
        },
        "image",
        "zarr",
    )
    
    # Remove the extension to get the base name
    filename_no_ext = filename_base.rsplit(".", 1)[0]
    zarr_path = base_path / f"{filename_no_ext}.zarr"
    tiff_path = base_path / f"{filename_no_ext}.tiff"
    
    if zarr_path.exists():
        sbs_image = read_image(str(zarr_path))
    elif tiff_path.exists():
        sbs_image = imread(str(tiff_path))
    else:
        pytest.fail(f"Neither Zarr nor TIFF found: {zarr_path} or {tiff_path}")
    
    assert sbs_image.shape == (5, 1200, 1200)


def test_convert_phenotype():
    # Try Zarr first (new default), then TIFF (legacy)
    base_path = PREPROCESS_FP / "images" / "phenotype"
    filename_base = get_filename(
        {"plate": TEST_PLATE, "well": TEST_WELL, "tile": TEST_TILE_PHENOTYPE},
        "image",
        "zarr",
    )
    
    # Remove the extension to get the base name
    filename_no_ext = filename_base.rsplit(".", 1)[0]
    zarr_path = base_path / f"{filename_no_ext}.zarr"
    tiff_path = base_path / f"{filename_no_ext}.tiff"
    
    if zarr_path.exists():
        phenotype_image = read_image(str(zarr_path))
    elif tiff_path.exists():
        phenotype_image = imread(str(tiff_path))
    else:
        pytest.fail(f"Neither Zarr nor TIFF found: {zarr_path} or {tiff_path}")
    
    assert phenotype_image.shape == (4, 2400, 2400)


def test_calculate_ic_sbs():
    # Try Zarr first (new default), then TIFF (legacy)
    base_path = PREPROCESS_FP / "ic_fields" / "sbs"
    filename_base = get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "cycle": TEST_CYCLE,
        },
        "ic_field",
        "zarr",
    )
    
    # Remove the extension to get the base name
    filename_no_ext = filename_base.rsplit(".", 1)[0]
    zarr_path = base_path / f"{filename_no_ext}.zarr"
    tiff_path = base_path / f"{filename_no_ext}.tiff"
    
    if zarr_path.exists():
        sbs_ic_field = read_image(str(zarr_path))
    elif tiff_path.exists():
        sbs_ic_field = imread(str(tiff_path))
    else:
        pytest.fail(f"Neither Zarr nor TIFF found: {zarr_path} or {tiff_path}")
    
    assert sbs_ic_field.shape == (5, 1200, 1200)


def test_calculate_ic_phenotype():
    # Try Zarr first (new default), then TIFF (legacy)
    base_path = PREPROCESS_FP / "ic_fields" / "phenotype"
    filename_base = get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
        },
        "ic_field",
        "zarr",
    )
    
    # Remove the extension to get the base name
    filename_no_ext = filename_base.rsplit(".", 1)[0]
    zarr_path = base_path / f"{filename_no_ext}.zarr"
    tiff_path = base_path / f"{filename_no_ext}.tiff"
    
    if zarr_path.exists():
        phenotype_ic_field = read_image(str(zarr_path))
    elif tiff_path.exists():
        phenotype_ic_field = imread(str(tiff_path))
    else:
        pytest.fail(f"Neither Zarr nor TIFF found: {zarr_path} or {tiff_path}")
    
    assert phenotype_ic_field.shape == (4, 2400, 2400)
