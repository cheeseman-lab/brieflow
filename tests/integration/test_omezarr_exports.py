
import pytest
import zarr
import os
from pathlib import Path

# Assuming tests are run from root or tests/small_test_analysis directory
# Adjust paths accordingly. Brieflow output is in brieflow_output/ relative to where snakemake ran.
# If running pytest from root, it might be tests/small_test_analysis/brieflow_output/

OUTPUT_DIR = Path("tests/small_test_analysis/brieflow_output")

@pytest.fixture(scope="module")
def check_output_dir():
    if not OUTPUT_DIR.exists():
        # Fallback if running from inside tests/small_test_analysis
        if Path("brieflow_output").exists():
            return Path("brieflow_output")
        pytest.skip("Brieflow output directory not found. Run run_brieflow_omezarr.sh first.")
    return OUTPUT_DIR

def test_preprocess_omezarr_exists(check_output_dir):
    # Expect: preprocess/omezarr/sbs/P-1_W-A1_T-001_C-1__image.zarr
    # We need to find at least one.
    zarr_dir = check_output_dir / "preprocess" / "omezarr"
    assert zarr_dir.exists()
    
    # Check SBS
    sbs_zarrs = list(zarr_dir.glob("sbs/*.zarr"))
    assert len(sbs_zarrs) > 0
    
    # Verify structure
    store = zarr.open(str(sbs_zarrs[0]), mode='r', zarr_format=2)
    assert "multiscales" in store.attrs
    assert "omero" in store.attrs
    assert "0" in store

def test_sbs_omezarr_exists(check_output_dir):
    zarr_dir = check_output_dir / "sbs" / "omezarr"
    assert zarr_dir.exists()
    
    zarrs = list(zarr_dir.glob("*.zarr"))
    assert len(zarrs) > 0
    
    store = zarr.open(str(zarrs[0]), mode='r', zarr_format=2)
    assert "multiscales" in store.attrs

def test_phenotype_omezarr_exists(check_output_dir):
    zarr_dir = check_output_dir / "phenotype" / "omezarr"
    assert zarr_dir.exists()
    
    # Images
    zarrs = list(zarr_dir.glob("*.zarr"))
    assert len(zarrs) > 0
    img_path = zarrs[0]
    
    # Labels
    nuclei_path = img_path / "labels" / "nuclei"
    cells_path = img_path / "labels" / "cells"
    
    assert nuclei_path.exists()
    assert cells_path.exists()
    
    # Check label metadata
    store = zarr.open(str(nuclei_path), mode='r', zarr_format=2)
    # OME-Zarr labels usually have image-label metadata or just be arrays
    # But write_labels_omezarr creates standard structure
    assert "0" in store

def test_merge_zarr_exists(check_output_dir):
    zarr_dir = check_output_dir / "merge" / "zarr"
    assert zarr_dir.exists()
    
    zarrs = list(zarr_dir.glob("*.zarr"))
    assert len(zarrs) > 0
    
    store = zarr.open(str(zarrs[0]), mode='r', zarr_format=2)
    assert "columns" in store.attrs

def test_aggregate_zarr_exists(check_output_dir):
    zarr_dir = check_output_dir / "aggregate" / "zarr"
    assert zarr_dir.exists()
    
    zarrs = list(zarr_dir.glob("*.zarr"))
    assert len(zarrs) > 0

def test_cluster_zarr_exists(check_output_dir):
    # Cluster zarrs are nested: cluster/zarr/combo/class/res/...
    zarr_dir = check_output_dir / "cluster"
    
    # Find any zarr file recursively
    zarrs = list(zarr_dir.glob("**/*.zarr"))
    assert len(zarrs) > 0
