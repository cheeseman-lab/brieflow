from pathlib import Path

import pytest


def test_extract_metadata_tile_nd2_includes_pixel_size_z_and_optics():
    nd2 = pytest.importorskip("nd2")

    from workflow.lib.preprocess.preprocess import extract_metadata_tile_nd2

    fp = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "small_test_analysis"
        / "small_test_data"
        / "phenotype"
        / "empty_images"
        / "P001_Pheno_20x_Wells-A1_Points-002__Channel_AF750,Cy3,GFP,DAPI.nd2"
    )
    assert fp.exists()

    df = extract_metadata_tile_nd2(
        file_path=str(fp),
        plate="1",
        well="A1",
        tile="2",
        verbose=False,
    )
    row = df.iloc[0]

    # These should now exist
    assert "pixel_size_z" in df.columns
    assert "objective_magnification" in df.columns
    assert "zoom_magnification" in df.columns
    assert "binning_xy" in df.columns

    # Values should be present for this ND2
    assert row["pixel_size_x"] is not None
    assert row["pixel_size_y"] is not None
    assert row["pixel_size_z"] is not None

    # Objective/zoom should be populated if present in metadata
    assert row["objective_magnification"] is not None
    assert row["zoom_magnification"] is not None


def test_convert_to_array_preserve_z_returns_czyx_for_tile_nd2():
    pytest.importorskip("nd2")

    from workflow.lib.preprocess.preprocess import convert_to_array

    fp = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "small_test_analysis"
        / "small_test_data"
        / "phenotype"
        / "empty_images"
        / "P001_Pheno_20x_Wells-A1_Points-002__Channel_AF750,Cy3,GFP,DAPI.nd2"
    )
    assert fp.exists()

    arr = convert_to_array(
        files=str(fp),
        data_format="nd2",
        data_organization="tile",
        preserve_z=True,
        verbose=False,
    )
    assert arr.ndim == 4  # CZYX
