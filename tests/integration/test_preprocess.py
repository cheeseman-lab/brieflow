import pytest
from pathlib import Path

import numpy as np
from tifffile import imread
import yaml

from lib.shared.file_utils import get_filename


TEST_ANALYSIS_PATH = Path("tests/data/small_test_analysis")
TEST_PLATE = 1
TEST_WELL = "A1"
TEST_TILE = 1

# load config file
CONFIG_FILE_PATH = TEST_ANALYSIS_PATH / "config/config.yml"
with open(CONFIG_FILE_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)


ROOT_FP = TEST_ANALYSIS_PATH / "analysis_root"
PREPROCESS_FP = ROOT_FP / "preprocess"


def test_convert_phenotype():
    phenotype_image_path = str(
        PREPROCESS_FP
        / "images"
        / "phenotype"
        / get_filename(
            {"plate": TEST_PLATE, "well": TEST_WELL, "tile": TEST_TILE},
            "image",
            "tiff",
        )
    )
    phenotype_image = imread(phenotype_image_path)
    assert phenotype_image.shape == (4, 2400, 2400)
