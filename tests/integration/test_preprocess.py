"""
Test analysis for the preprocess module.
"""

from pathlib import Path

import yaml
import pandas as pd
from tifffile import imread

from lib.shared.file_utils import get_filename

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


ROOT_FP = TEST_ANALYSIS_PATH / "brieflow_output"
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
    sbs_image_path = str(
        PREPROCESS_FP
        / "images"
        / "sbs"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
                "tile": TEST_TILE_SBS,
                "cycle": TEST_CYCLE,
            },
            "image",
            "tiff",
        ),
    )
    sbs_image = imread(sbs_image_path)
    assert sbs_image.shape == (5, 1200, 1200)


def test_convert_phenotype():
    phenotype_image_path = str(
        PREPROCESS_FP
        / "images"
        / "phenotype"
        / get_filename(
            {"plate": TEST_PLATE, "well": TEST_WELL, "tile": TEST_TILE_PHENOTYPE},
            "image",
            "tiff",
        )
    )
    phenotype_image = imread(phenotype_image_path)
    assert phenotype_image.shape == (4, 2400, 2400)


def test_calculate_ic_sbs():
    sbs_ic_field_path = str(
        PREPROCESS_FP
        / "ic_fields"
        / "sbs"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
                "cycle": TEST_CYCLE,
            },
            "ic_field",
            "tiff",
        ),
    )
    sbs_ic_field = imread(sbs_ic_field_path)
    assert sbs_ic_field.shape == (5, 1200, 1200)


def test_calculate_ic_phenotype():
    phenotype_ic_field_path = str(
        PREPROCESS_FP
        / "ic_fields"
        / "phenotype"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
            },
            "ic_field",
            "tiff",
        ),
    )
    phenotype_ic_field = imread(phenotype_ic_field_path)
    assert phenotype_ic_field.shape == (4, 2400, 2400)
