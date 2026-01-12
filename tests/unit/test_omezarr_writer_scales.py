import json
import numpy as np


def _read_zattrs(zarr_path):
    return json.loads((zarr_path / ".zattrs").read_text())


def test_write_image_omezarr_scalar_pixel_size_sets_xy_scale(tmp_path):
    from workflow.lib.io.omezarr_writer import write_image_omezarr

    out = tmp_path / "img.zarr"
    # Use a reasonably sized image to avoid edge cases in pyramid generation
    img = np.zeros((1, 256, 256), dtype=np.uint16)  # cyx
    write_image_omezarr(img, str(out), axes="cyx", pixel_size_um=0.325)

    zattrs = _read_zattrs(out)
    scale0 = zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    assert scale0 == [1.0, 0.325, 0.325]


def test_write_image_omezarr_dict_pixel_size_sets_xyz_scale(tmp_path):
    from workflow.lib.io.omezarr_writer import write_image_omezarr

    out = tmp_path / "img3d.zarr"
    img = np.zeros((1, 2, 128, 128), dtype=np.uint16)  # czyx
    write_image_omezarr(
        img,
        str(out),
        axes="czyx",
        pixel_size_um={"z": 1.5, "y": 0.325, "x": 0.325},
    )

    zattrs = _read_zattrs(out)
    scale0 = zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    assert scale0 == [1.0, 1.5, 0.325, 0.325]


def test_write_image_omezarr_xyz_scale_does_not_downsample_z(tmp_path):
    """
    ome-zarr-py generates pyramids by downsampling X/Y only; Z resolution stays constant.
    """
    from workflow.lib.io.omezarr_writer import write_image_omezarr

    out = tmp_path / "img3d.zarr"
    img = np.zeros((1, 4, 128, 128), dtype=np.uint16)  # czyx
    write_image_omezarr(
        img,
        str(out),
        axes="czyx",
        pixel_size_um={"z": 1.5, "y": 0.325, "x": 0.325},
        coarsening_factor=2,
        max_levels=2,
    )

    zattrs = _read_zattrs(out)
    scale0 = zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ]
    scale1 = zattrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0][
        "scale"
    ]

    assert scale0 == [1.0, 1.5, 0.325, 0.325]
    assert scale1 == [1.0, 1.5, 0.65, 0.65]
