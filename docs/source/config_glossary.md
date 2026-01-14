# Config Glossary

The brieflow config holds all of the parameters used for a brieflow run.
Each notebook is used to configure the parameter variables, which are then saved to the `config.yml`.
Each analysis requires a specific `config.yml` and the associated files (pool dataframe, cell classification model, etc).
An example config for the `main` branch is outlined in [tests/small_test_analysis/config/config.yml](https://github.com/cheeseman-lab/brieflow/blob/main/tests/small_test_analysis/config/config.yml).
While all of the parameters are explicity outlined in each notebook, we provide additional comments on some here as well:
- `preprocess:sbs_samples_fp`/`preprocess:phenotype_samples_fp`: Path to dataframes with one entry for an SBS/phenotype file's path and the associated metadata (plate, well, tile, etc).
- `sbs:df_design_path`: Path to dataframe with SBS pool design information regarding gene, sgRNA, oligo, etc.
- `*_combo_fp`: Path to dataframe with wildcards for file processing in a particular module.
Each combination usually corresponds to a one process that needs to be done with Snakemake.
For example, each plate, well, tile combination in `phenotype_combo_fp` corresponds to one ND2 -> tiff file conversion during preprocessing.

## Input Data Configuration

### `preprocess:sbs_data_format` and `preprocess:phenotype_data_format`

Specifies the format of your input microscopy files.

**Options:**
- `"nd2"` - Nikon ND2 format (default)
- `"tiff"` - TIFF format

**Example:**
```yaml
preprocess:
  sbs_data_format: "nd2"
  phenotype_data_format: "nd2"
```

These parameters tell the workflow how to read your source files. ND2 is Nikon's proprietary microscopy format, while TIFF is a standard image format.

### `preprocess:sbs_data_organization` and `preprocess:phenotype_data_organization`

Specifies how your input files are organized.

**Options:**
- `"tile"` - Each file contains a single field of view (FOV/tile)
- `"well"` - Each file contains all tiles for a well

**Example:**
```yaml
preprocess:
  sbs_data_organization: "tile"
  phenotype_data_organization: "tile"
```

**Note:** When `data_format: "tiff"`, the workflow automatically uses `"tile"` organization regardless of this setting.

## Zarr Support

Brieflow supports both TIFF and Zarr formats for image storage and processing. Zarr is a cloud-native format that offers better performance for large datasets and enables visualization in tools like Napari.

**Note:** All intermediate image processing steps (alignment, filtering, segmentation) use TIFF format. Zarr is used for:
- Initial ND2 → image conversion (preprocessing)
- Final visualization outputs (OME-Zarr format)

### Configuration Overview

Zarr support is controlled by two independent config sections:

1. **`preprocess:output_formats`** - Controls preprocessing outputs (ND2 conversion)
2. **`output:omezarr`** - Controls OME-Zarr visualization exports at any pipeline stage

### Preprocessing Zarr Outputs

#### `preprocess:output_formats`

**Default:** `["zarr"]` (Zarr is used by default if not specified)

**Options:**
- `"zarr"` - Creates Zarr outputs during preprocessing
- `"tiff"` - Creates TIFF outputs during preprocessing
- `["tiff", "zarr"]` - Creates both formats

When `"zarr"` is enabled, preprocessing creates:
- **Standard Zarr arrays** in `preprocess/images/sbs/` and `preprocess/images/phenotype/` (if used for downstream processing)
- **OME-Zarr multiscale** in `preprocess/omezarr/sbs/` and `preprocess/omezarr/phenotype/` (always created for visualization)

#### `preprocess:downstream_input_format`

**Default:** `"tiff"` if TIFF is enabled, otherwise `"zarr"`

Controls which format the SBS and phenotype modules use as input:
- `"tiff"` - Use TIFF files for downstream processing
- `"zarr"` - Use standard Zarr arrays for downstream processing

**Note:** Illumination correction (IC) fields are saved in the same format as `downstream_input_format`.

#### Example Configurations

**Default behavior (no config needed):**
```yaml
# Uses Zarr for preprocessing and downstream processing
# Creates OME-Zarr for visualization
```

**TIFF for processing, Zarr for visualization:**
```yaml
preprocess:
  output_formats: ["tiff", "zarr"]
  downstream_input_format: "tiff"
```

### OME-Zarr Visualization Exports

#### `output:omezarr`

Controls OME-Zarr exports at different pipeline stages. These are separate from preprocessing Zarr outputs.

**Parameters:**
- `enabled` - Enable/disable OME-Zarr exports (default: `false`)
- `after_steps` - List of pipeline stages to export
- `layout` - Layout format (default: `"per_image"`)

**Available steps:**
- `"sbs"` - Aligned SBS cycles with segmentation masks
- `"phenotype"` - Processed phenotype images with segmentation masks
- `"merge"` - Merged SBS and phenotype data
- `"aggregate"` - Aggregated features
- `"cluster"` - Clustering results

**Example (from config_omezarr.yml):**
```yaml
output:
  omezarr:
    enabled: true
    after_steps: ["preprocess", "sbs", "phenotype", "merge", "aggregate", "cluster"]
    layout: "per_image"
```

**Note:** Including `"preprocess"` in `after_steps` is optional—preprocessing OME-Zarr outputs are created automatically when `output_formats` includes `"zarr"`.

### Output Directory Structure

```
brieflow_output/
└── preprocess/
    ├── images/
    │   ├── sbs/
    │   │   ├── P-1_W-A1_T-0_C-1__image.zarr/    # Standard Zarr (if downstream_input_format: "zarr")
    │   │   └── P-1_W-A1_T-0_C-1__image.tiff     # TIFF (if output_formats includes "tiff")
    │   └── phenotype/
    │       └── P-1_W-A1_T-5__image.zarr/
    ├── omezarr/
    │   ├── sbs/
    │   │   └── P-1_W-A1_T-0_C-1__image.zarr/    # OME-Zarr multiscale (for visualization)
    │   │       ├── 0/  # Full resolution
    │   │       ├── 1/  # 2x downsampled
    │   │       ├── 2/  # 4x downsampled
    │   │       └── 3/  # 8x downsampled
    │   └── phenotype/
    └── ic_fields/
        └── sbs/
            └── P-1_W-A1_C-1__ic_field.zarr/     # IC field (matches downstream_input_format)
```

### Visualizing OME-Zarr Outputs

Load OME-Zarr files in Napari using the provided scripts:

**Command-line:**
```bash
python workflow/scripts/shared/load_omezarr_in_napari.py /path/to/file.zarr
```

**Jupyter notebook:**
```python
# In workflow/scripts/shared/load_omezarr_notebook.py
zarr_path = "/path/to/your/file.zarr"
# Run the script
```

The scripts automatically handle:
- Multiscale image pyramids
- Pixel scaling and axes metadata
- Segmentation mask labels
- Channel-specific colormaps
- Optimal visualization settings
