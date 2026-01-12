# OME-Zarr Compliance Tests

This directory contains unit tests to ensure all zarr outputs from the workflow comply with the [OME-Zarr specification](https://ngff.openmicroscopy.org/latest/).

## Overview

The tests use the official [ome-zarr-py](https://ome-zarr.readthedocs.io/en/stable/api.html) library to validate:
- Format detection and version compliance
- Multiscales metadata structure
- Pyramid level generation
- Coordinate transformations and scaling
- Channel metadata (OMERO format)
- Label/segmentation image compliance
- Data integrity (round-trip read/write)

## Installation

Install test dependencies:

```bash
pip install pytest ome-zarr zarr numpy
```

## Running Tests

### Run all tests:
```bash
cd /Users/cspeters/projects/Brieflow/workflow/tests
pytest test_omezarr_compliance.py -v
```

### Run specific test class:
```bash
pytest test_omezarr_compliance.py::TestOMEZarrCompliance -v
```

### Run specific test:
```bash
pytest test_omezarr_compliance.py::TestOMEZarrCompliance::test_format_detection -v
```

### Generate coverage report:
```bash
pytest test_omezarr_compliance.py --cov=../lib/shared/omezarr_io --cov-report=html
```

## Test Structure

### `TestOMEZarrCompliance`
Main test suite covering:
- **Format Detection**: Validates that outputs are recognized as OME-Zarr
- **Reader Compatibility**: Ensures ome-zarr Reader can open files
- **Multiscale Metadata**: Checks required fields (version, axes, datasets)
- **Pyramid Levels**: Verifies multi-resolution pyramid generation
- **Scale Consistency**: Validates coordinate transformations
- **4D Images**: Tests CZYX format with Z-stacks
- **Label Images**: Validates segmentation mask compliance
- **Channel Names**: Checks OMERO channel metadata
- **Data Integrity**: Round-trip read/write validation
- **Chunk Alignment**: Verifies zarr chunking
- **Dimension Separator**: Ensures '/' separator usage

### `TestOMEZarrEdgeCases`
Edge case testing:
- Single-level pyramids
- Small images
- Various data types

## Key Compliance Checks

### 1. Format Detection
```python
from ome_zarr.format import detect_format
detected_format = detect_format(zarr_path)
assert detected_format is not None
```

### 2. Multiscales Metadata
Required fields according to OME-Zarr spec:
- `version`: OME-Zarr version string
- `axes`: Array of axis objects with `name` and `type`
- `datasets`: Array of dataset objects with `path` and `coordinateTransformations`

### 3. Coordinate Transformations
Each dataset must have:
- `type`: "scale"
- `scale`: Array of scaling factors for each axis

### 4. OMERO Metadata (Optional but Recommended)
- `channels`: Array of channel objects with `label` and `color`

## What Gets Tested

| Component | Test Coverage |
|-----------|---------------|
| `write_multiscale_omezarr()` | ✅ Full |
| `save_image()` wrapper | ✅ Full |
| 3D images (CYX) | ✅ Full |
| 4D images (CZYX) | ✅ Full |
| Label images | ✅ Full |
| Channel names | ✅ Full |
| Pyramid generation | ✅ Full |
| Scale transformations | ✅ Full |
| Chunk alignment | ✅ Full |
| Data integrity | ✅ Full |

## Integration with Workflow

These tests validate the core zarr writing functions used throughout the workflow:

### Preprocessing
- `nd2_to_omezarr()` → Uses `write_multiscale_omezarr()`
- SBS and phenotype image conversion

### Segmentation
- `save_image()` for nuclei/cell masks
- Label image generation

### Illumination Correction
- IC field generation as zarr

## Continuous Integration

Add to CI pipeline:

```yaml
- name: Test OME-Zarr Compliance
  run: |
    pip install pytest ome-zarr zarr numpy
    pytest workflow/tests/test_omezarr_compliance.py -v --junitxml=test-results.xml
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'ome_zarr'**
```bash
pip install ome-zarr
```

**Format Not Detected**
- Check that .zattrs file exists and contains `multiscales` field
- Verify JSON structure matches OME-Zarr spec
- Ensure all required metadata fields are present

**Reader Cannot Open Zarr**
- Verify zarr format version (should be 2)
- Check dimension_separator is '/'
- Ensure all pyramid levels have .zarray files

## References

- [OME-Zarr Specification](https://ngff.openmicroscopy.org/latest/)
- [ome-zarr-py Documentation](https://ome-zarr.readthedocs.io/en/stable/)
- [ome-zarr-py API Reference](https://ome-zarr.readthedocs.io/en/stable/api.html)
- [Zarr Specification](https://zarr.readthedocs.io/en/stable/spec/v2.html)

## Contributing

When adding new zarr writing functionality:
1. Add corresponding tests to `test_omezarr_compliance.py`
2. Run tests locally before committing
3. Ensure all tests pass in CI

## License

Same as parent project.

