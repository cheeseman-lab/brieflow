
import shutil
import tempfile
import numpy as np
import pandas as pd
import zarr
import pytest
from pathlib import Path
from workflow.lib.io.omezarr_writer import write_image_omezarr, write_labels_omezarr, write_table_zarr

@pytest.fixture
def temp_zarr_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.ome.zarr"

def test_write_image_roundtrip(temp_zarr_path):
    # Generate random (C, Y, X) data
    shape = (3, 64, 64)
    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    
    write_image_omezarr(
        image_data=data,
        out_path=str(temp_zarr_path),
        channel_names=["r", "g", "b"],
        axes="cyx"
    )
    
    # Read back
    store = zarr.open(str(temp_zarr_path), mode='r')
    
    # Check multiscales metadata
    assert "multiscales" in store.attrs
    
    # Check level 0 array
    level0 = store['0'][:]
    np.testing.assert_array_equal(data, level0)
    
    # Check channel metadata
    omero_meta = store.attrs.get("omero")
    assert omero_meta is not None
    assert len(omero_meta["channels"]) == 3
    assert omero_meta["channels"][0]["label"] == "r"

def test_write_labels(temp_zarr_path):
    # Create an image first
    image_shape = (64, 64)
    image_data = np.zeros(image_shape, dtype=np.uint8)
    write_image_omezarr(image_data, str(temp_zarr_path), axes="yx")
    
    # Create random labels
    labels = np.random.randint(0, 5, size=image_shape, dtype=np.uint32)
    
    write_labels_omezarr(
        label_data=labels,
        out_path=str(temp_zarr_path),
        label_name="nuclei",
        axes="yx"
    )
    
    # Verify labels exist
    store = zarr.open(str(temp_zarr_path), mode='r')
    assert "labels" in store
    assert "nuclei" in store["labels"]
    
    # Read back labels
    label_arr = store["labels"]["nuclei"]["0"][:]
    np.testing.assert_array_equal(labels, label_arr)

def test_write_table(temp_zarr_path):
    # Create simple dataframe
    df = pd.DataFrame({
        "cell_id": [1, 2, 3],
        "score": [0.1, 0.5, 0.9],
        "class": ["A", "B", "A"]
    })
    
    table_path = str(temp_zarr_path).replace(".ome.zarr", ".zarr")
    write_table_zarr(df, table_path)
    
    # Read back manually
    store = zarr.open(table_path, mode='r')
    assert "cell_id" in store
    assert "score" in store
    assert "class" in store
    
    np.testing.assert_array_equal(store["cell_id"][:], df["cell_id"].values)
    np.testing.assert_array_almost_equal(store["score"][:], df["score"].values)
    
    # Check string decoding if necessary or just array equality
    read_classes = store["class"][:]
    assert np.all(read_classes == df["class"].values)
