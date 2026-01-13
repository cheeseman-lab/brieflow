#!/usr/bin/env python3
"""Compare preprocessed images between zarr and tiff workflows."""

import sys
import numpy as np
from pathlib import Path

# Add workflow lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "workflow"))

from lib.shared.io import read_image

# Paths to compare
zarr_preprocess = "/Users/cspeters/projects/Brieflow/tests/small_test_analysis/brieflow_output/preprocess/omezarr/phenotype/P-1_W-A1_T-2__image.zarr"
tiff_preprocess = "/Users/cspeters/brieflow/tests/small_test_analysis/brieflow_output/preprocess/images/phenotype/P-1_W-A1_T-2__image.tiff"

zarr_ic_corrected = "/Users/cspeters/projects/Brieflow/tests/small_test_analysis/brieflow_output/phenotype/images/P-1_W-A1_T-2__illumination_corrected.zarr"
tiff_ic_corrected = "/Users/cspeters/brieflow/tests/small_test_analysis/brieflow_output/phenotype/images/P-1_W-A1_T-2__illumination_corrected.tiff"

print("=" * 80)
print("COMPARING PREPROCESSED IMAGES (before IC)")
print("=" * 80)

# Read zarr preprocessed
print(f"\nReading zarr: {zarr_preprocess}")
zarr_data = read_image(zarr_preprocess)
print(f"  Shape: {zarr_data.shape}")
print(f"  Dtype: {zarr_data.dtype}")
print(f"  Min: {zarr_data.min()}, Max: {zarr_data.max()}, Mean: {zarr_data.mean():.2f}")
print(
    f"  First channel stats: min={zarr_data[0].min()}, max={zarr_data[0].max()}, mean={zarr_data[0].mean():.2f}"
)

# Read tiff preprocessed
print(f"\nReading tiff: {tiff_preprocess}")
tiff_data = read_image(tiff_preprocess)
print(f"  Shape: {tiff_data.shape}")
print(f"  Dtype: {tiff_data.dtype}")
print(f"  Min: {tiff_data.min()}, Max: {tiff_data.max()}, Mean: {tiff_data.mean():.2f}")
print(
    f"  First channel stats: min={tiff_data[0].min()}, max={tiff_data[0].max()}, mean={tiff_data[0].mean():.2f}"
)

# Compare
if zarr_data.shape == tiff_data.shape:
    print(f"\n✓ Shapes match: {zarr_data.shape}")
    if np.array_equal(zarr_data, tiff_data):
        print("✓ Arrays are IDENTICAL")
    else:
        diff = np.abs(zarr_data.astype(float) - tiff_data.astype(float))
        print(f"✗ Arrays DIFFER:")
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.4f}")
        print(
            f"  Num different pixels: {np.sum(diff > 0)} / {diff.size} ({100 * np.sum(diff > 0) / diff.size:.2f}%)"
        )
else:
    print(f"\n✗ Shapes DO NOT match: zarr={zarr_data.shape} vs tiff={tiff_data.shape}")

print("\n" + "=" * 80)
print("COMPARING IC-CORRECTED IMAGES")
print("=" * 80)

# Read zarr IC corrected
print(f"\nReading zarr IC: {zarr_ic_corrected}")
zarr_ic_data = read_image(zarr_ic_corrected)
print(f"  Shape: {zarr_ic_data.shape}")
print(f"  Dtype: {zarr_ic_data.dtype}")
print(
    f"  Min: {zarr_ic_data.min()}, Max: {zarr_ic_data.max()}, Mean: {zarr_ic_data.mean():.2f}"
)
print(
    f"  First channel stats: min={zarr_ic_data[0].min()}, max={zarr_ic_data[0].max()}, mean={zarr_ic_data[0].mean():.2f}"
)

# Read tiff IC corrected
print(f"\nReading tiff IC: {tiff_ic_corrected}")
tiff_ic_data = read_image(tiff_ic_corrected)
print(f"  Shape: {tiff_ic_data.shape}")
print(f"  Dtype: {tiff_ic_data.dtype}")
print(
    f"  Min: {tiff_ic_data.min()}, Max: {tiff_ic_data.max()}, Mean: {tiff_ic_data.mean():.2f}"
)
print(
    f"  First channel stats: min={tiff_ic_data[0].min()}, max={tiff_ic_data[0].max()}, mean={tiff_ic_data[0].mean():.2f}"
)

# Compare
if zarr_ic_data.shape == tiff_ic_data.shape:
    print(f"\n✓ Shapes match: {zarr_ic_data.shape}")
    if np.array_equal(zarr_ic_data, tiff_ic_data):
        print("✓ Arrays are IDENTICAL")
    else:
        diff = np.abs(zarr_ic_data.astype(float) - tiff_ic_data.astype(float))
        print(f"✗ Arrays DIFFER:")
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.4f}")
        print(
            f"  Num different pixels: {np.sum(diff > 0)} / {diff.size} ({100 * np.sum(diff > 0) / diff.size:.2f}%)"
        )
else:
    print(
        f"\n✗ Shapes DO NOT match: zarr={zarr_ic_data.shape} vs tiff={tiff_ic_data.shape}"
    )

print("\n" + "=" * 80)
print("COMPARING IC FIELDS")
print("=" * 80)

zarr_ic_field = "/Users/cspeters/projects/Brieflow/tests/small_test_analysis/brieflow_output/preprocess/ic_fields/phenotype/P-1_W-A1__ic_field.zarr"
tiff_ic_field = "/Users/cspeters/brieflow/tests/small_test_analysis/brieflow_output/preprocess/ic_fields/phenotype/P-1_W-A1__ic_field.tiff"

print(f"\nReading zarr IC field: {zarr_ic_field}")
zarr_ic_field_data = read_image(zarr_ic_field)
print(f"  Shape: {zarr_ic_field_data.shape}")
print(f"  Dtype: {zarr_ic_field_data.dtype}")
print(
    f"  Min: {zarr_ic_field_data.min()}, Max: {zarr_ic_field_data.max()}, Mean: {zarr_ic_field_data.mean():.2f}"
)

print(f"\nReading tiff IC field: {tiff_ic_field}")
tiff_ic_field_data = read_image(tiff_ic_field)
print(f"  Shape: {tiff_ic_field_data.shape}")
print(f"  Dtype: {tiff_ic_field_data.dtype}")
print(
    f"  Min: {tiff_ic_field_data.min()}, Max: {tiff_ic_field_data.max()}, Mean: {tiff_ic_field_data.mean():.2f}"
)

# Compare IC fields
if zarr_ic_field_data.shape == tiff_ic_field_data.shape:
    print(f"\n✓ IC field shapes match: {zarr_ic_field_data.shape}")
    if np.array_equal(zarr_ic_field_data, tiff_ic_field_data):
        print("✓ IC fields are IDENTICAL")
    else:
        diff = np.abs(
            zarr_ic_field_data.astype(float) - tiff_ic_field_data.astype(float)
        )
        print(f"✗ IC fields DIFFER:")
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.4f}")
        print(
            f"  Num different pixels: {np.sum(diff > 0)} / {diff.size} ({100 * np.sum(diff > 0) / diff.size:.2f}%)"
        )
else:
    print(
        f"\n✗ IC field shapes DO NOT match: zarr={zarr_ic_field_data.shape} vs tiff={tiff_ic_field_data.shape}"
    )
