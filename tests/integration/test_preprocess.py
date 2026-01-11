"""
Integration tests for the preprocess module.
"""

from pathlib import Path

import yaml
import pandas as pd
from tifffile import imread

from lib.shared.file_utils import get_filename
from lib.shared.io import read_image

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
    assert sbs_metadata.shape[1] == 12


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
    assert phenotype_metadata.shape[1] == 11


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
    # Get the configured downstream input format (tiff or zarr)
    downstream_format = config["preprocess"].get("downstream_input_format", "tiff")
    
    sbs_ic_field_path = (
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
            downstream_format,
        )
    )
    
    # Use read_image which handles both tiff and zarr
    sbs_ic_field = read_image(sbs_ic_field_path)
    assert sbs_ic_field.shape == (5, 1200, 1200)


def test_calculate_ic_phenotype():
    # Get the configured downstream input format (tiff or zarr)
    downstream_format = config["preprocess"].get("downstream_input_format", "tiff")
    
    phenotype_ic_field_path = (
        PREPROCESS_FP
        / "ic_fields"
        / "phenotype"
        / get_filename(
            {
                "plate": TEST_PLATE,
                "well": TEST_WELL,
            },
            "ic_field",
            downstream_format,
        )
    )
    
    # Use read_image which handles both tiff and zarr
    phenotype_ic_field = read_image(phenotype_ic_field_path)
    
    # Phenotype images can be CYX or CZYX depending on whether they have Z-stacks
    # The shape should be (4 channels, ..., 2400, 2400)
    assert phenotype_ic_field.shape[0] == 4  # 4 channels
    assert phenotype_ic_field.shape[-2:] == (2400, 2400)  # Y, X dimensions
    assert phenotype_ic_field.ndim in (3, 4)  # CYX or CZYX
