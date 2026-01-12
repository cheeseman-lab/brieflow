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

## Zarr Support

Brieflow supports both TIFF and Zarr formats for image storage and processing. Zarr is a cloud-native format that offers better performance for large datasets and enables visualization in tools like Napari.

### Preprocessing Output Formats

The `preprocess` section of the config file supports the following parameters for controlling output formats:

#### `output_formats`
Controls which file formats are created during preprocessing. Can be a single string or a list.

**Options:**
- `"tiff"`: Create TIFF files in `preprocess/images/sbs/` and `preprocess/images/phenotype/`
- `"zarr"`: Create both:
  - Standard Zarr arrays in `preprocess/images/sbs/` and `preprocess/images/phenotype/` (for downstream processing)
  - OME-Zarr multiscale images in `preprocess/omezarr/sbs/` and `preprocess/omezarr/phenotype/` (for visualization)

**Default:** `["zarr"]`

**Examples:**
```yaml
preprocess:
  output_formats: "zarr"  # Zarr only
```

```yaml
preprocess:
  output_formats: ["tiff", "zarr"]  # Both formats
```

```yaml
preprocess:
  output_formats: "tiff"  # TIFF only (legacy)
```

#### `downstream_input_format`
Controls which format is used for downstream SBS and phenotype analysis modules.

**Options:**
- `"tiff"`: Use TIFF files for downstream processing
- `"zarr"`: Use standard Zarr arrays for downstream processing (NOT OME-Zarr multiscale)

**Default:** Automatically set to `"tiff"` if TIFF is enabled in `output_formats`, otherwise `"zarr"`

**Example:**
```yaml
preprocess:
  output_formats: ["tiff", "zarr"]
  downstream_input_format: "zarr"  # Use Zarr for downstream processing
```

**Note:** The illumination correction (IC) fields will be saved in the same format as `downstream_input_format`.

### OME-Zarr Visualization Outputs

The `output` section controls when OME-Zarr visualization outputs are created for various pipeline stages.

#### `output:omezarr`
Configuration for OME-Zarr exports at different pipeline stages.

**Parameters:**
- `enabled`: Boolean to enable/disable OME-Zarr exports (default: `false`)
- `after_steps`: List of pipeline stages after which to create OME-Zarr outputs
- `layout`: Layout format for OME-Zarr files (default: `"per_image"`)

**Available steps:**
- `"preprocess"`: Raw converted images with IC fields applied
- `"sbs"`: Aligned SBS cycles with all processing steps
- `"phenotype"`: Processed phenotype images with segmentation masks
- `"merge"`: Merged SBS and phenotype data
- `"aggregate"`: Aggregated features
- `"cluster"`: Clustering results

**Example:**
```yaml
output:
  omezarr:
    enabled: true
    after_steps: ["preprocess", "sbs", "phenotype"]
    layout: "per_image"
```

### Zarr Format Details

**Standard Zarr Arrays:**
- Used for downstream processing in SBS and phenotype modules
- Stored in `preprocess/images/sbs/` and `preprocess/images/phenotype/`
- Single-scale, optimized for computational processing
- Compatible with all Brieflow processing scripts via `lib.shared.io.read_image()`

**OME-Zarr Multiscale:**
- Used for visualization in Napari and other OME-NGFF compatible viewers
- Stored in `preprocess/omezarr/sbs/` and `preprocess/omezarr/phenotype/`
- Multiscale pyramids for efficient visualization at different zoom levels
- Compliant with OME-NGFF v0.4 specification
- Includes metadata for pixel sizes, channel names, and axes information

### Visualizing OME-Zarr Outputs

Brieflow provides scripts to load OME-Zarr files into Napari for visualization:

**Command-line script:**
```bash
python workflow/scripts/shared/load_omezarr_in_napari.py /path/to/file.zarr
```

**Jupyter notebook script:**
```python
# In workflow/scripts/shared/load_omezarr_notebook.py
zarr_path = "/path/to/your/file.zarr"
# Run the script
```

Both scripts automatically:
- Load multiscale image data
- Apply correct pixel scales
- Load associated label layers (segmentation masks)
- Set appropriate colormaps for each channel
- Configure visualization settings (interpolation, contrast limits)
