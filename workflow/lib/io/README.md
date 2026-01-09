# OME-Zarr Export Specification (Design A)

This document defines the specification for optional OME-Zarr v2 exports in the Brieflow pipeline.

## Overview

Brieflow maintains TIFF-based intermediates for its computational workflow. When `output.omezarr.enabled` is True in the configuration, the pipeline additionally exports data to OME-Zarr (Zarr v2) format after key milestones.

## Configuration

The export is controlled via the `output.omezarr` block in `config.yml`:

```yaml
output:
  omezarr:
    enabled: true
    after_steps: ["preprocess", "sbs", "phenotype", "merge", "aggregate", "cluster"]
    layout: "per_image"  # Future support for "plate"
    write_args:
      compressor: "default"
```

## Milestone Exports

### 1. Preprocess
*   **Source:** Tiled TIFF images from `preprocess/images/{type}/...`.
*   **Export Path:** `preprocess/omezarr/{well}_{tile}__{type}.zarr`
*   **Structure:**
    *   Root group attrs: `multiscales`, `omero` (channel info).
    *   Arrays: `0`, `1`, `2`... (pyramid levels).
    *   Axes: `(c, y, x)`.

### 2. SBS
*   **Source:** Aligned SBS images (cycles/channels).
*   **Export Path:** `sbs/omezarr/{well}_{tile}__sbs_aligned.zarr`
*   **Structure:**
    *   Axes: `(cycle, channel, y, x)` or `(c, y, x)` flattened.
    *   Metadata: Channel names include cycle info (e.g., `C1-A`, `C1-C`).

### 3. Phenotype
*   **Source:** Aligned phenotype images and segmentation masks.
*   **Export Path (Images):** `phenotype/omezarr/{well}_{tile}__phenotype.zarr`
*   **Export Path (Labels):** `phenotype/omezarr/{well}_{tile}__phenotype.zarr/labels/{label_name}`
    *   `labels/nuclei`
    *   `labels/cells`
*   **Structure:**
    *   Images: `(c, y, x)`.
    *   Labels: `(y, x)` integer arrays linked to image via `image-label` metadata.

### 4. Merge
*   **Source:** Merged cell tables (Parquet).
*   **Export Path:** `merge/zarr/{well}__merge_table.zarr`
*   **Structure:**
    *   Zarr group containing arrays for each column.
    *   Simple schema metadata.

### 5. Aggregate & Cluster
*   **Source:** Aggregated feature tables.
*   **Export Path:** `aggregate/zarr/{class}_{combo}__aggregate.zarr`, `cluster/zarr/...`
*   **Structure:**
    *   Columnar arrays.

## Technical Details

*   **Format:** OME-Zarr v0.4 (Zarr v2).
*   **Pyramids:** Power-of-2 downsampling (mean for images, nearest for labels).
*   **Chunking:** Default to `(1, 1024, 1024)` or similar for images.
*   **Compression:** Blosc Zstd (default).
