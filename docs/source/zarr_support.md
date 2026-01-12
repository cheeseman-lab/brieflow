# Zarr Support in Brieflow

Brieflow supports Zarr, a cloud-native array storage format, as an alternative to TIFF files. Zarr offers significant advantages for large-scale microscopy data processing and visualization.

## Overview

Zarr is a format for storing chunked, compressed, N-dimensional arrays. In Brieflow, Zarr support includes:

1. **Standard Zarr arrays** for efficient downstream processing
2. **OME-Zarr multiscale images** for interactive visualization in Napari and other OME-NGFF compatible viewers

## Benefits of Using Zarr

### Performance
- **Faster I/O**: Chunked storage enables parallel read/write operations
- **Efficient compression**: Blosc compression (zstd codec) reduces storage requirements
- **Lazy loading**: Only load data chunks as needed, reducing memory usage

### Scalability
- **Cloud-native**: Designed for distributed computing and cloud storage
- **Large datasets**: Handles multi-terabyte datasets efficiently
- **Parallel processing**: Multiple workers can read/write simultaneously

### Visualization
- **Multiscale pyramids**: OME-Zarr creates resolution pyramids for smooth zooming
- **Interactive viewing**: Napari and other viewers can stream data without loading entire images
- **Metadata preservation**: Pixel sizes, channel names, and axes information embedded in the format

## Configuration

### Basic Setup

Add the following to your `config.yml` to enable Zarr:

```yaml
preprocess:
  output_formats: "zarr"
```

This will:
- Create standard Zarr arrays in `preprocess/images/sbs/` and `preprocess/images/phenotype/`
- Create OME-Zarr multiscale images in `preprocess/omezarr/sbs/` and `preprocess/omezarr/phenotype/`
- Use Zarr for downstream SBS and phenotype processing

### Dual Format Support

To create both TIFF and Zarr outputs:

```yaml
preprocess:
  output_formats: ["tiff", "zarr"]
  downstream_input_format: "tiff"  # or "zarr"
```

This is useful for:
- Transitioning from TIFF to Zarr workflows
- Compatibility with external tools that require TIFF
- Comparing performance between formats

### OME-Zarr Visualization Outputs

Enable OME-Zarr exports at various pipeline stages:

```yaml
output:
  omezarr:
    enabled: true
    after_steps: ["preprocess", "sbs", "phenotype", "merge"]
    layout: "per_image"
```

**Available steps:**
- `preprocess`: Raw converted images with IC fields applied
- `sbs`: Aligned SBS cycles with all processing steps
- `phenotype`: Processed phenotype images with segmentation masks
- `merge`: Merged SBS and phenotype data
- `aggregate`: Aggregated features
- `cluster`: Clustering results

## File Structure

### Standard Zarr Arrays

```
brieflow_output/
└── preprocess/
    ├── images/
    │   ├── sbs/
    │   │   └── P-1_W-A1_T-0_C-1__image.zarr/
    │   │       ├── .zarray
    │   │       ├── .zattrs
    │   │       └── [data chunks]
    │   └── phenotype/
    │       └── P-1_W-A1_T-5__image.zarr/
    └── ic_fields/
        ├── sbs/
        │   └── P-1_W-A1_C-1__ic_field.zarr/
        └── phenotype/
            └── P-1_W-A1__ic_field.zarr/
```

### OME-Zarr Multiscale

```
brieflow_output/
└── preprocess/
    └── omezarr/
        ├── sbs/
        │   └── P-1_W-A1_T-0_C-1__image.zarr/
        │       ├── .zattrs          # OME-NGFF metadata
        │       ├── .zgroup
        │       ├── 0/               # Full resolution
        │       │   ├── .zarray
        │       │   └── [chunks]
        │       ├── 1/               # 2x downsampled
        │       ├── 2/               # 4x downsampled
        │       └── 3/               # 8x downsampled
        └── phenotype/
            └── P-1_W-A1_T-5__image.zarr/
```

## Visualization with Napari

Brieflow provides scripts to visualize OME-Zarr outputs in Napari.

### Command-Line Script

```bash
python workflow/scripts/shared/load_omezarr_in_napari.py /path/to/file.zarr
```

**Features:**
- Automatically loads all resolution levels
- Applies correct pixel scales from metadata
- Loads associated label layers (segmentation masks)
- Sets channel-specific colormaps
- Configures optimal visualization settings

### Jupyter Notebook Script

For interactive exploration in Jupyter:

```python
# In a Jupyter notebook
%run workflow/scripts/shared/load_omezarr_notebook.py

# Or modify the script directly:
zarr_path = "/path/to/your/file.zarr"
# Then run the script
```

### Example: Loading SBS Data

```bash
# Load aligned SBS cycles
python workflow/scripts/shared/load_omezarr_in_napari.py \
  brieflow_output/sbs/omezarr/P-1_W-A1_T-0__sbs_aligned.zarr
```

This will open Napari with:
- All SBS cycles as separate time points
- All channels (DAPI, G, T, A, C) with distinct colormaps
- Correct pixel scaling
- Segmentation masks if available

### Example: Loading Phenotype Data

```bash
# Load phenotype image with segmentation
python workflow/scripts/shared/load_omezarr_in_napari.py \
  brieflow_output/phenotype/omezarr/P-1_W-A1_T-5__phenotype.zarr
```

This will display:
- All phenotype channels with appropriate colormaps
- Nuclei and cell segmentation masks as label layers
- Proper spatial scaling

## Technical Details

### Zarr Format Specifications

**Standard Zarr Arrays:**
- Format: Zarr v2 (with v3 compatibility)
- Compression: Blosc with zstd codec (level 3)
- Chunk size: Optimized based on image dimensions
- Data type: Preserved from source (typically uint16)
- Axes order: CYX (Channel, Y, X) or CZYX (Channel, Z, Y, X)

**OME-Zarr Multiscale:**
- Specification: OME-NGFF v0.4
- Pyramid levels: 4 levels (1x, 2x, 4x, 8x downsampling)
- Downsampling method: Mean pooling
- Metadata: Includes axes, scales, channel names, and units
- Compression: Same as standard Zarr

### Compatibility

**Reading Zarr in Brieflow:**
All Brieflow processing scripts use `lib.shared.io.read_image()`, which automatically detects and reads both TIFF and Zarr formats:

```python
from lib.shared.io import read_image

# Works with both TIFF and Zarr
image = read_image("path/to/image.zarr")  # or .tiff
```

**External Tools:**
- **Napari**: Native support via ome-zarr plugin
- **Python**: Read with `zarr` or `dask` libraries
- **ImageJ/Fiji**: Via Bio-Formats (limited support)
- **MATLAB**: Via zarr-matlab library

## Performance Considerations

### When to Use Zarr

**Recommended for:**
- Large screens (>100 GB of image data)
- Cloud-based or distributed computing
- Interactive visualization during analysis
- Long-term data storage and sharing

**TIFF may be sufficient for:**
- Small pilot screens (<50 GB)
- Local processing on high-performance workstations
- Compatibility with legacy tools

### Storage Requirements

Zarr typically uses **30-50% less storage** than uncompressed TIFF due to Blosc compression. However, OME-Zarr multiscale adds ~33% overhead for pyramid levels.

**Example for a typical screen:**
- Raw ND2 files: 500 GB
- TIFF outputs: 450 GB
- Standard Zarr: 250 GB
- OME-Zarr (with pyramids): 330 GB

### Processing Speed

Zarr can be **2-5x faster** than TIFF for:
- Illumination correction calculation (parallel chunk processing)
- Large image alignment (lazy loading of regions)
- Feature extraction (selective chunk reading)

## Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'zarr'"**
```bash
# Install zarr package
pip install zarr
```

**Issue: Napari can't open OME-Zarr files**
```bash
# Install ome-zarr plugin
pip install ome-zarr napari[all]
```

**Issue: Zarr files are larger than expected**
- Check compression settings in the Zarr metadata (`.zattrs`)
- Ensure Blosc is installed: `pip install zarr[blosc]`
- Verify chunk size is appropriate for your data

**Issue: Slow Zarr reading**
- Increase chunk cache size: `zarr.open(..., cache_size=1e9)`
- Use SSD storage for Zarr files
- Check network latency if using cloud storage

### Validation

To verify OME-Zarr compliance:

```python
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# Open and validate
store = parse_url("path/to/file.zarr", mode="r")
reader = Reader(store)
nodes = list(reader())

# Check metadata
for node in nodes:
    print(node.metadata)
```

## Migration Guide

### Transitioning from TIFF to Zarr

1. **Test with a small dataset:**
   ```yaml
   preprocess:
     output_formats: ["tiff", "zarr"]
     sample_fraction: 0.1  # Process 10% of data
   ```

2. **Compare outputs:**
   - Verify image data matches between formats
   - Check processing times
   - Validate downstream results

3. **Switch to Zarr-only:**
   ```yaml
   preprocess:
     output_formats: "zarr"
   ```

4. **Update workflows:**
   - All Brieflow scripts automatically support Zarr
   - External scripts may need `read_image()` wrapper

### Converting Existing TIFF to Zarr

```python
from lib.shared.io import read_image, save_image_zarr
import zarr

# Read TIFF
image = read_image("image.tiff")

# Save as Zarr
root = zarr.open_group("image.zarr", mode='w')
# ... (see workflow/scripts/preprocess/nd2_to_zarr.py for full example)
```

## Additional Resources

- [Zarr Documentation](https://zarr.readthedocs.io/)
- [OME-NGFF Specification](https://ngff.openmicroscopy.org/)
- [Napari Documentation](https://napari.org/)
- [Brieflow GitHub Repository](https://github.com/cheeseman-lab/brieflow)

## Example Configurations

### Minimal Zarr Setup
```yaml
preprocess:
  output_formats: "zarr"
  # All other settings remain the same
```

### Complete Zarr Configuration
```yaml
preprocess:
  output_formats: ["tiff", "zarr"]
  downstream_input_format: "zarr"
  sbs_data_format: "nd2"
  phenotype_data_format: "nd2"
  # ... other preprocess settings

output:
  omezarr:
    enabled: true
    after_steps: ["preprocess", "sbs", "phenotype"]
    layout: "per_image"
```

### Zarr-Only Production Setup
```yaml
preprocess:
  output_formats: "zarr"
  # Downstream automatically uses Zarr

output:
  omezarr:
    enabled: true
    after_steps: ["sbs", "phenotype", "merge"]
    layout: "per_image"
```

