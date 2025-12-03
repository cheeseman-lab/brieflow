# Brieflow Architecture Analysis and OME-Zarr Export Plan

**Date:** December 2, 2025  
**Purpose:** Comprehensive documentation of Brieflow architecture, testing strategy, and OME-Zarr export implementation plan

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Module Dependencies & Data Flow](#3-module-dependencies--data-flow)
4. [Existing Pipeline Testing Plan](#4-existing-pipeline-testing-plan)
5. [OME-Zarr Export Implementation Plan](#5-ome-zarr-export-implementation-plan)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Project Overview

### What is Brieflow?

Brieflow is an extensible Snakemake-based pipeline for processing optical pooled screening (OPS) data. It combines:
- **Sequencing by Synthesis (SBS)** data for genetic perturbation identification
- **Phenotype imaging** data for cellular morphology and features
- **Computational analysis** for cell matching, feature extraction, and clustering

### Key Technologies

- **Snakemake**: Workflow orchestration
- **Python 3.11**: Core language
- **ND2 Format**: Nikon microscopy input format
- **TIFF**: Current intermediate format
- **OME-Zarr**: Target format for scalable image storage
- **CellProfiler/cp_measure**: Feature extraction
- **Cellpose/StarDist/MicroSAM**: Cell segmentation
- **PHATE + Leiden**: Dimensionality reduction and clustering

---

## 2. Architecture & File Structure

### 2.1 Top-Level Structure

```
Brieflow/
├── workflow/
│   ├── lib/              # Brieflow library (core processing functions)
│   ├── rules/            # Snakemake rule definitions
│   ├── scripts/          # Python scripts called by rules
│   ├── targets/          # Input/output mapping definitions
│   └── Snakefile         # Main workflow entry point
├── tests/
│   ├── integration/      # Integration tests
│   └── small_test_analysis/  # End-to-end test data
├── docs/                 # Documentation
├── visualization/        # Visualization utilities
├── tasks/               # Task tracking
└── pyproject.toml       # Python package configuration
```

### 2.2 Library Organization (`workflow/lib/`)

The library code is organized by processing module:

#### **preprocess/** - Data conversion and illumination correction
- `preprocess.py`: ND2 → TIFF/OME-Zarr conversion, metadata extraction
- `file_utils.py`: File path management and sample handling

#### **sbs/** - Sequencing by Synthesis processing
- `align_cycles.py`: Register sequencing cycles
- `find_peaks.py`: Detect sequencing spots (using Spotiflow or standard detection)
- `extract_bases.py`: Extract base intensity at each spot
- `call_reads.py`: Call sequencing reads with cross-talk correction
- `call_cells.py`: Map reads to cells using barcode library
- `max_filter.py`: Dilate channels for alignment tolerance
- `compute_standard_deviation.py`: Quality metric computation
- Pre-trained models: `basecaller.joblib`, `segmented_bc.joblib`, etc.

#### **phenotype/** - Phenotype imaging processing
- `align_channels.py`: Register phenotype imaging rounds
- `extract_phenotype_cp_measure.py`: Feature extraction using cp_measure
- `extract_phenotype_cp_multichannel.py`: Multi-channel feature extraction
- `identify_cytoplasm_cellpose.py`: Cytoplasm identification
- `eval_features.py`: Feature quality assessment

#### **merge/** - SBS + Phenotype integration
- `hash.py`: Triangle hashing for coarse alignment
- `merge.py`: Fine alignment using linear regression
- `format_merge.py`: Clean and format merged data
- `deduplicate_merge.py`: Remove duplicate cell mappings
- `eval_merge.py`: Merge quality metrics

#### **aggregate/** - Multi-well data aggregation
- `aggregate.py`: Aggregate replicate measurements
- `align.py`: Batch effect correction (PCA-based)
- `filter.py`: Quality filtering (missing values, outliers)
- `cell_classification.py`: Mitotic/interphase classification
- `montage_utils.py`: Generate cell montages for visualization

#### **cluster/** - Phenotype clustering and analysis
- `phate_leiden_clustering.py`: PHATE embedding + Leiden clustering
- `cluster_analysis.py`: Differential feature analysis
- `benchmark_clusters.py`: Validation against STRING/CORUM/KEGG

#### **shared/** - Common utilities
- `image_utils.py`: Image processing utilities
- `illumination_correction.py`: Illumination field calculation
- `segment_cellpose.py`: Cellpose segmentation wrapper
- `segment_stardist.py`: StarDist segmentation wrapper
- `segment_microsam.py`: MicroSAM segmentation wrapper
- `segment_watershed.py`: Watershed segmentation
- `feature_extraction.py`: Feature extraction utilities
- `align.py`: Image registration utilities
- `target_utils.py`: Snakemake wildcard/target utilities
- `rule_utils.py`: Parameter extraction for rules
- `eval_segmentation.py`: Segmentation quality metrics

#### **external/** - Third-party code
- `cp_emulator.py`: CellProfiler emulation functions

### 2.3 Snakemake Rules Organization (`workflow/rules/`)

Each module has a corresponding `.smk` file defining its rules:

- **preprocess.smk**: ND2 conversion, metadata extraction, illumination correction
- **sbs.smk**: Complete SBS pipeline (align → segment → call reads → call cells)
- **phenotype.smk**: Phenotype pipeline (align → segment → extract features)
- **merge.smk**: Integration pipeline (align → merge → deduplicate)
- **aggregate.smk**: Aggregation pipeline (filter → align → aggregate → montage)
- **cluster.smk**: Clustering pipeline (clean → cluster → benchmark)

### 2.4 Scripts Organization (`workflow/scripts/`)

Scripts mirror the rule structure and call library functions:

```
scripts/
├── preprocess/
│   ├── nd2_to_tiff.py
│   ├── nd2_to_omezarr.py
│   ├── extract_tile_metadata.py
│   └── calculate_ic_field.py
├── sbs/
│   ├── align_cycles.py
│   ├── find_peaks.py
│   ├── extract_bases.py
│   ├── call_reads.py
│   └── call_cells.py
├── phenotype/
│   ├── align_phenotype.py
│   ├── extract_phenotype_cp_multichannel.py
│   └── identify_cytoplasm_cellpose.py
├── merge/
│   ├── fast_alignment.py
│   ├── merge.py
│   ├── format_merge.py
│   └── deduplicate_merge.py
├── aggregate/
│   ├── split_datasets.py
│   ├── filter.py
│   ├── align.py
│   └── aggregate.py
├── cluster/
│   ├── phate_leiden_clustering.py
│   └── benchmark_clusters.py
└── shared/
    ├── segment.py
    ├── extract_phenotype_minimal.py
    └── combine_dfs.py
```

### 2.5 Configuration System

Configuration is managed through YAML files:

```yaml
all:
  root_fp: brieflow_output/  # Output directory

preprocess:
  sbs_samples_fp: config/sbs_samples.tsv
  sbs_combo_fp: config/sbs_combo.tsv
  phenotype_samples_fp: config/phenotype_samples.tsv
  phenotype_combo_fp: config/phenotype_combo.tsv
  omezarr_chunk_shape: [1, 512, 512]
  omezarr_coarsening_factor: 2
  omezarr_max_levels: null

sbs:
  channel_names: [DAPI, G, T, A, C]
  segmentation_method: cellpose  # or stardist, microsam, watershed
  spot_detection_method: standard  # or spotiflow
  # ... many more parameters

phenotype:
  channel_names: [DAPI, COXIV, CENPA, WGA]
  segmentation_method: cellpose
  # ... more parameters

merge:
  det_range: [0.06, 0.065]
  threshold: 2

aggregate:
  agg_method: median
  classifier_path: config/binary_xgb_robust_model.dill

cluster:
  phate_distance_metric: cosine
  leiden_resolutions: [8, 10, 12]
```

---

## 3. Module Dependencies & Data Flow

### 3.1 Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         PREPROCESS                              │
│  ND2 Files → Metadata Extraction → TIFF/OME-Zarr Conversion   │
│              → Illumination Correction                          │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │                       │
        ┌─────────────▼─────────┐   ┌────────▼──────────┐
        │        SBS            │   │     PHENOTYPE      │
        │  • Align cycles       │   │  • Align channels  │
        │  • Find peaks         │   │  • Segment cells   │
        │  • Extract bases      │   │  • Extract         │
        │  • Call reads         │   │    features (CP)   │
        │  • Segment cells      │   │  • Evaluate        │
        │  • Call cells         │   │    segmentation    │
        │  • Evaluate mapping   │   │                    │
        └─────────────┬─────────┘   └────────┬───────────┘
                      │                      │
                      └──────────┬───────────┘
                                 │
                      ┌──────────▼──────────┐
                      │       MERGE         │
                      │  • Triangle hashing │
                      │  • Fine alignment   │
                      │  • Format & clean   │
                      │  • Deduplicate      │
                      │  • Evaluate merge   │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │     AGGREGATE       │
                      │  • Split by class   │
                      │  • Filter outliers  │
                      │  • Batch correction │
                      │  • Aggregate wells  │
                      │  • Generate montage │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │      CLUSTER        │
                      │  • Clean data       │
                      │  • PHATE embedding  │
                      │  • Leiden clustering│
                      │  • Benchmark        │
                      └─────────────────────┘
```

### 3.2 Key Function Call Chains

#### **Preprocess Module**
```
Rule: convert_sbs / convert_phenotype
  ↓
Script: nd2_to_tiff.py / nd2_to_omezarr.py
  ↓
Library: lib.preprocess.preprocess.nd2_to_tiff()
  or     lib.preprocess.preprocess.nd2_to_omezarr()
  ↓
Library: lib.preprocess.preprocess.write_multiscale_omezarr()
```

#### **SBS Module**
```
Rule: align_sbs
  ↓
Script: scripts/sbs/align_cycles.py
  ↓
Library: lib.sbs.align_cycles.align()
  ↓
Library: lib.shared.align.apply_shifts()

Rule: find_peaks
  ↓
Script: scripts/sbs/find_peaks.py
  ↓
Library: lib.sbs.find_peaks.find_peaks_spotiflow()
  or     lib.sbs.find_peaks.find_peaks()

Rule: call_reads
  ↓
Script: scripts/sbs/call_reads.py
  ↓
Library: lib.sbs.call_reads.call_reads()
```

#### **Phenotype Module**
```
Rule: segment_phenotype
  ↓
Script: scripts/shared/segment.py
  ↓
Library: lib.shared.segment_cellpose.segment_cellpose()
  or     lib.shared.segment_stardist.segment_stardist()
  or     lib.shared.segment_microsam.segment_microsam()

Rule: extract_phenotype_cp
  ↓
Script: scripts/phenotype/extract_phenotype_cp_multichannel.py
  ↓
Library: lib.phenotype.extract_phenotype_cp_multichannel.extract_phenotype_cp_multichannel()
  ↓
Library: lib.external.cp_emulator.extract_features()
```

#### **Merge Module**
```
Rule: fast_alignment
  ↓
Script: scripts/merge/fast_alignment.py
  ↓
Library: lib.merge.hash.multistep_alignment()

Rule: merge
  ↓
Script: scripts/merge/merge.py
  ↓
Library: lib.merge.merge.merge_triangle_hash()
  ↓
Library: lib.merge.merge.merge_sbs_phenotype()
```

#### **Aggregate Module**
```
Rule: filter
  ↓
Script: scripts/aggregate/filter.py
  ↓
Library: lib.aggregate.filter.query_filter()
         lib.aggregate.filter.missing_values_filter()
         lib.aggregate.filter.outlier_filter()

Rule: align
  ↓
Script: scripts/aggregate/align.py
  ↓
Library: lib.aggregate.align.prepare_alignment_data()
```

#### **Cluster Module**
```
Rule: phate_leiden_clustering
  ↓
Script: scripts/cluster/phate_leiden_clustering.py
  ↓
Library: lib.cluster.phate_leiden_clustering.phate_leiden_clustering()
  ↓
External: phate.PHATE(), leidenalg
```

### 3.3 Data Format Transformations

```
Input: ND2 (Nikon microscopy format)
  ↓
Preprocess: TIFF (CYX format, uint16) + Metadata TSV
           OR OME-Zarr (multiscale pyramid)
  ↓
SBS: Parquet tables (reads, cells, bases)
     + Segmentation masks (TIFF, labels)
  ↓
Phenotype: Parquet tables (features)
          + Segmentation masks (TIFF, labels)
  ↓
Merge: Parquet tables (merged cell identities)
  ↓
Aggregate: Parquet tables (aggregated features)
          + PNG montages
  ↓
Cluster: Parquet tables (cluster assignments, embeddings)
        + PNG plots
```

### 3.4 Critical Dependencies Between Modules

1. **SBS** and **Phenotype** depend on **Preprocess**:
   - Require converted images (TIFF or OME-Zarr)
   - Require illumination correction fields
   - Require metadata tables

2. **Merge** depends on **SBS** and **Phenotype**:
   - Requires cell location tables from both
   - Requires segmentation outputs
   - Requires barcode calls from SBS

3. **Aggregate** depends on **Merge**:
   - Requires merged cell-feature tables
   - Requires cell classification (mitotic/interphase)

4. **Cluster** depends on **Aggregate**:
   - Requires aggregated feature tables
   - Requires batch-corrected embeddings

---

## 4. Existing Pipeline Testing Plan

### 4.1 Test Environment Setup

#### Prerequisites
```bash
# 1. Create conda environment
conda create -n brieflow_test -c conda-forge python=3.11 uv pip -y
conda activate brieflow_test

# 2. Install dependencies
uv pip install -r pyproject.toml
uv pip install -e .

# 3. Install conda-only packages (optional)
conda install -c conda-forge micro_sam -y
```

#### Test Data Preparation
```bash
# Navigate to test directory
cd tests/small_test_analysis

# Download and extract test data (~200MB)
python small_test_analysis_setup.py

# Expected structure after setup:
# small_test_analysis/
# ├── config/
# │   ├── config.yml
# │   ├── sbs_samples.tsv
# │   ├── phenotype_samples.tsv
# │   └── barcode_library.tsv
# └── small_test_data/
#     ├── sbs/
#     │   └── *.nd2 files
#     └── phenotype/
#         └── *.nd2 files
```

### 4.2 Module-by-Module Testing Strategy

#### **Test 1: Preprocess Module**

**Objective**: Validate ND2 → TIFF/OME-Zarr conversion and illumination correction

```bash
# Run only preprocess
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_preprocess

# Verify outputs
ls brieflow_output/preprocess/metadata/sbs/
ls brieflow_output/preprocess/images/sbs/
ls brieflow_output/preprocess/ic_fields/sbs/
```

**Expected Outputs**:
- Metadata TSV files for each tile/cycle/well
- Combined metadata parquet files per well
- TIFF images (C×Y×X, uint16) for each tile
- Illumination correction fields per cycle/well

**Validation Checks**:
```python
# Run integration tests
pytest tests/integration/test_preprocess.py

# Manual validation
from tifffile import imread
import pandas as pd

# Check image dimensions
img = imread("brieflow_output/preprocess/images/sbs/P-1_W-A1_T-0_C-11__image.tiff")
assert img.shape == (5, 1200, 1200)  # 5 channels, 1200×1200 pixels

# Check metadata
meta = pd.read_parquet("brieflow_output/preprocess/metadata/sbs/P-1_W-A1__combined_metadata.parquet")
assert "x_pos" in meta.columns
assert "y_pos" in meta.columns
```

#### **Test 2: SBS Module**

**Objective**: Validate sequencing read calling and cell barcode mapping

```bash
# Run SBS pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_sbs

# Verify outputs
ls brieflow_output/sbs/aligned/
ls brieflow_output/sbs/segmentation/
ls brieflow_output/sbs/reads/
ls brieflow_output/sbs/cells/
```

**Expected Outputs**:
- Aligned SBS images (cycles registered)
- Segmentation masks (nuclei, cells)
- Reads table (parquet): read ID, position, barcode, quality scores
- Cells table (parquet): cell ID, barcode, counts

**Validation Checks**:
```python
import pandas as pd

# Check reads table
reads = pd.read_parquet("brieflow_output/sbs/reads/P-1_W-A1__reads.parquet")
assert "barcode" in reads.columns
assert "Q_min" in reads.columns
assert len(reads) > 0

# Check cells table
cells = pd.read_parquet("brieflow_output/sbs/cells/P-1_W-A1__cells.parquet")
assert "sgRNA" in cells.columns or "sgRNA_0" in cells.columns
assert "count" in cells.columns

# Check mapping rate
mapping_rate = len(cells) / len(reads)
print(f"Mapping rate: {mapping_rate:.2%}")
assert mapping_rate > 0.1  # At least 10% of reads should map
```

#### **Test 3: Phenotype Module**

**Objective**: Validate cell segmentation and feature extraction

```bash
# Run phenotype pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_phenotype

# Verify outputs
ls brieflow_output/phenotype/segmentation/
ls brieflow_output/phenotype/features/
```

**Expected Outputs**:
- Aligned phenotype images
- Segmentation masks (nuclei, cells, cytoplasm)
- Feature tables (parquet): morphology, intensity, texture features

**Validation Checks**:
```python
import pandas as pd

# Check feature table
features = pd.read_parquet("brieflow_output/phenotype/features/P-1_W-A1__features_cp.parquet")
assert "nucleus_DAPI_int_mean" in features.columns or any("intensity" in col for col in features.columns)
assert "label" in features.columns
assert len(features) > 0

# Check feature coverage
feature_cols = [col for col in features.columns if col not in ["label", "plate", "well", "tile"]]
print(f"Number of features extracted: {len(feature_cols)}")
assert len(feature_cols) > 50  # Should have many morphology features
```

#### **Test 4: Merge Module**

**Objective**: Validate SBS-Phenotype cell matching

```bash
# Run merge pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_merge

# Verify outputs
ls brieflow_output/merge/alignment/
ls brieflow_output/merge/merged/
```

**Expected Outputs**:
- Alignment parameters (rotation, translation)
- Merged cell tables with SBS + phenotype features
- Deduplication statistics

**Validation Checks**:
```python
import pandas as pd

# Check merge quality
merge = pd.read_parquet("brieflow_output/merge/merged/P-1_W-A1__merged.parquet")
assert "cell_0" in merge.columns  # Phenotype cell ID
assert "cell_1" in merge.columns  # SBS cell ID
assert "distance" in merge.columns  # Matching distance

# Check merge rate
sbs_cells = pd.read_parquet("brieflow_output/sbs/cells/P-1_W-A1__cells.parquet")
merge_rate = len(merge) / len(sbs_cells)
print(f"Merge rate: {merge_rate:.2%}")
assert merge_rate > 0.5  # At least 50% should merge
```

#### **Test 5: Aggregate Module**

**Objective**: Validate multi-well aggregation and batch correction

```bash
# Run aggregate pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_aggregate

# Verify outputs
ls brieflow_output/aggregate/filtered/
ls brieflow_output/aggregate/aligned/
ls brieflow_output/aggregate/aggregated/
```

**Expected Outputs**:
- Filtered cell-level data
- Batch-corrected embeddings
- Aggregated gene-level profiles
- Cell montages (PNG)

**Validation Checks**:
```python
import pandas as pd

# Check aggregated data
agg = pd.read_parquet("brieflow_output/aggregate/aggregated/all_all__aggregated.parquet")
assert "gene_symbol_0" in agg.columns or "perturbation_name" in agg.columns
assert "cell_count" in agg.columns

# Check number of genes
n_genes = agg["gene_symbol_0"].nunique()
print(f"Number of genes: {n_genes}")
assert n_genes > 10  # Should have multiple genes
```

#### **Test 6: Cluster Module**

**Objective**: Validate clustering and benchmark analysis

```bash
# Run cluster pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_cluster

# Verify outputs
ls brieflow_output/cluster/embeddings/
ls brieflow_output/cluster/clusters/
ls brieflow_output/cluster/benchmarks/
```

**Expected Outputs**:
- PHATE embeddings (parquet)
- Cluster assignments (parquet)
- Benchmark scores vs. STRING/CORUM/KEGG

**Validation Checks**:
```python
import pandas as pd

# Check clustering
clusters = pd.read_parquet("brieflow_output/cluster/clusters/all_8__clusters.parquet")
assert "cluster" in clusters.columns
assert "gene_symbol_0" in clusters.columns

# Check cluster distribution
cluster_counts = clusters["cluster"].value_counts()
print(f"Number of clusters: {len(cluster_counts)}")
print(cluster_counts)
```

### 4.3 End-to-End Integration Test

```bash
# Run complete pipeline
cd tests/small_test_analysis
sh run_brieflow.sh

# This runs all modules in sequence:
# all_preprocess → all_sbs → all_phenotype → all_merge → all_aggregate → all_cluster
```

**Expected Runtime**: ~14 minutes on a standard workstation

### 4.4 Automated Testing with Pytest

```bash
# Run all integration tests
cd /path/to/Brieflow
pytest tests/integration/

# Run specific test module
pytest tests/integration/test_preprocess.py
pytest tests/integration/test_omezarr_writer.py

# Run with verbose output
pytest -v tests/

# Generate coverage report
pytest --cov=lib --cov-report=html tests/
```

### 4.5 Validation Checklist

After running the full pipeline, verify:

- [ ] All Snakemake rules completed without errors
- [ ] No missing output files
- [ ] Image dimensions are correct (check TIFF/OME-Zarr shapes)
- [ ] Metadata tables have expected columns
- [ ] Cell counts are reasonable (>100 cells per well)
- [ ] Mapping rates are acceptable (>10% reads → cells, >50% SBS → phenotype)
- [ ] Feature extraction produced >50 features per cell
- [ ] Aggregation produced gene-level profiles
- [ ] Clustering identified multiple clusters
- [ ] Evaluation plots generated successfully

---

## 5. OME-Zarr Export Implementation Plan

### 5.1 Why OME-Zarr?

**Current Limitations with TIFF**:
- Single-resolution storage (no pyramid)
- Limited metadata support
- Not optimized for cloud/remote access
- Difficult to visualize large images in browsers

**OME-Zarr Advantages**:
- **Multi-scale pyramids**: Fast zooming at any level
- **Chunked storage**: Efficient partial reads
- **Cloud-native**: S3/GCS compatible
- **Standardized metadata**: OME-NGFF v0.4 specification
- **Visualization support**: Napari, Vizarr, Neuroglancer

### 5.2 Existing OME-Zarr Infrastructure

Brieflow **already has** OME-Zarr writing capabilities implemented:

#### **Library Functions** (in `lib/preprocess/preprocess.py`):
1. `write_multiscale_omezarr()`: Core writer function
2. `nd2_to_omezarr()`: ND2 → OME-Zarr converter
3. `_build_pyramid()`: Create multi-scale levels
4. `_write_zarr_array()`: Write individual pyramid levels
5. `_write_group_metadata()`: Write OME-Zarr metadata

#### **Snakemake Rules** (in `workflow/rules/preprocess.smk`):
- `convert_sbs_omezarr` (lines 97-116)
- `convert_phenotype_omezarr` (lines 137-156)

#### **Test Coverage**:
- `tests/integration/test_omezarr_writer.py`: Unit tests for OME-Zarr writer

### 5.3 Current OME-Zarr Implementation Status

**What Works**:
✅ OME-Zarr writing for preprocess module  
✅ Multi-scale pyramid generation  
✅ Chunked storage with configurable chunk size  
✅ OME-NGFF v0.4 metadata  
✅ Pixel size metadata from ND2 files  
✅ Unit tests for core functionality  

**What's Missing**:
❌ OME-Zarr export after **SBS** module processing  
❌ OME-Zarr export after **Phenotype** module processing  
❌ OME-Zarr export after **Merge** module (if desired)  
❌ OME-Zarr export for segmentation masks  
❌ Integration into downstream analysis workflows  
❌ Documentation for OME-Zarr workflows  

### 5.4 Detailed Implementation Plan

#### **Phase 1: Enable OME-Zarr After SBS Module**

**Goal**: Export aligned, filtered SBS images as OME-Zarr

**Outputs to Export**:
1. Aligned SBS images (`sbs/aligned/`)
2. Log-filtered images (`sbs/log_filtered/`)
3. Max-filtered images (`sbs/max_filtered/`)
4. IC-corrected images (`sbs/ic_corrected/`)

**Implementation Steps**:

1. **Add target definitions** in `workflow/targets/sbs.smk`:
```python
"align_sbs_omezarr": [
    SBS_FP
    / "omezarr"
    / "aligned"
    / get_filename(
        {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
        "aligned",
        "ome.zarr",
    ),
],
"log_filter_omezarr": [
    SBS_FP
    / "omezarr"
    / "log_filtered"
    / get_filename(
        {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
        "log_filtered",
        "ome.zarr",
    ),
],
```

2. **Create rule** in `workflow/rules/sbs.smk`:
```python
rule export_align_sbs_omezarr:
    input:
        SBS_OUTPUTS["align_sbs"],
    output:
        SBS_OUTPUTS_MAPPED["align_sbs_omezarr"],
    params:
        chunk_shape=config["sbs"].get("omezarr_chunk_shape", [1, 512, 512]),
        coarsening_factor=config["sbs"].get("omezarr_coarsening_factor", 2),
        max_levels=config["sbs"].get("omezarr_max_levels"),
    script:
        "../scripts/sbs/export_aligned_omezarr.py"
```

3. **Create script** `workflow/scripts/sbs/export_aligned_omezarr.py`:
```python
from pathlib import Path
from tifffile import imread
from lib.preprocess.preprocess import write_multiscale_omezarr

# Read TIFF input
image = imread(snakemake.input[0])

# Write OME-Zarr
write_multiscale_omezarr(
    image=image,
    output_dir=Path(snakemake.output[0]),
    pixel_size=(1.0, 1.0),  # Or extract from metadata
    chunk_shape=tuple(snakemake.params.chunk_shape),
    coarsening_factor=snakemake.params.coarsening_factor,
    max_levels=snakemake.params.max_levels,
)
```

4. **Update configuration** in `config/config.yml`:
```yaml
sbs:
  # ... existing parameters ...
  omezarr_export: true  # Enable OME-Zarr export
  omezarr_chunk_shape: [1, 512, 512]
  omezarr_coarsening_factor: 2
  omezarr_max_levels: 5
  omezarr_stages:  # Which processing stages to export
    - aligned
    - log_filtered
    - ic_corrected
```

5. **Add conditional export** in `workflow/rules/sbs.smk`:
```python
if config["sbs"].get("omezarr_export", False):
    SBS_TARGETS_ALL.extend(SBS_TARGETS_OMEZARR)
```

#### **Phase 2: Enable OME-Zarr After Phenotype Module**

**Goal**: Export aligned, segmented phenotype images as OME-Zarr

**Outputs to Export**:
1. IC-corrected images (`phenotype/ic_corrected/`)
2. Aligned images (`phenotype/aligned/`)
3. Segmentation masks (nuclei, cells, cytoplasm)

**Implementation Steps**:

1. **Add target definitions** in `workflow/targets/phenotype.smk`:
```python
"align_phenotype_omezarr": [
    PHENOTYPE_FP
    / "omezarr"
    / "aligned"
    / get_filename(
        {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
        "aligned",
        "ome.zarr",
    ),
],
"segment_phenotype_omezarr": [
    PHENOTYPE_FP
    / "omezarr"
    / "segmentation"
    / get_filename(
        {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
        "segmentation",
        "ome.zarr",
    ),
],
```

2. **Create rules** in `workflow/rules/phenotype.smk`:
```python
rule export_align_phenotype_omezarr:
    input:
        PHENOTYPE_OUTPUTS["align_phenotype"],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["align_phenotype_omezarr"],
    params:
        chunk_shape=config["phenotype"].get("omezarr_chunk_shape", [1, 512, 512]),
        coarsening_factor=config["phenotype"].get("omezarr_coarsening_factor", 2),
        max_levels=config["phenotype"].get("omezarr_max_levels"),
    script:
        "../scripts/phenotype/export_aligned_omezarr.py"

rule export_segment_phenotype_omezarr:
    input:
        nuclei=PHENOTYPE_OUTPUTS["segment_phenotype"][0],
        cells=PHENOTYPE_OUTPUTS["segment_phenotype"][1],
    output:
        PHENOTYPE_OUTPUTS_MAPPED["segment_phenotype_omezarr"],
    params:
        chunk_shape=config["phenotype"].get("omezarr_chunk_shape", [1, 512, 512]),
        coarsening_factor=config["phenotype"].get("omezarr_coarsening_factor", 2),
        max_levels=config["phenotype"].get("omezarr_max_levels"),
    script:
        "../scripts/phenotype/export_segmentation_omezarr.py"
```

3. **Create script** `workflow/scripts/phenotype/export_segmentation_omezarr.py`:
```python
import numpy as np
from pathlib import Path
from tifffile import imread
from lib.preprocess.preprocess import write_multiscale_omezarr

# Read segmentation masks
nuclei = imread(snakemake.input.nuclei)
cells = imread(snakemake.input.cells)

# Stack into multi-channel array (2 channels)
segmentation = np.stack([nuclei, cells], axis=0)

# Write OME-Zarr with label metadata
write_multiscale_omezarr(
    image=segmentation,
    output_dir=Path(snakemake.output[0]),
    pixel_size=(1.0, 1.0),
    chunk_shape=tuple(snakemake.params.chunk_shape),
    coarsening_factor=snakemake.params.coarsening_factor,
    max_levels=snakemake.params.max_levels,
)

# Add label-specific metadata
import json
attrs_path = Path(snakemake.output[0]) / ".zattrs"
attrs = json.loads(attrs_path.read_text())
attrs["omero"]["channels"][0]["label"] = "Nuclei"
attrs["omero"]["channels"][1]["label"] = "Cells"
attrs_path.write_text(json.dumps(attrs, indent=2))
```

#### **Phase 3: Create Library Function for Generic OME-Zarr Export**

**Goal**: Reusable function for exporting any TIFF to OME-Zarr

**New Library Function** in `lib/shared/omezarr_utils.py`:

```python
"""Utilities for OME-Zarr export throughout the pipeline."""

from pathlib import Path
from typing import Union, Sequence, Optional
import numpy as np
from tifffile import imread
from lib.preprocess.preprocess import write_multiscale_omezarr


def export_tiff_to_omezarr(
    tiff_path: Union[str, Path],
    output_dir: Union[str, Path],
    pixel_size: tuple[float, float] = (1.0, 1.0),
    chunk_shape: Sequence[int] = (1, 512, 512),
    coarsening_factor: int = 2,
    max_levels: Optional[int] = None,
    channel_labels: Optional[list[str]] = None,
    is_label_image: bool = False,
) -> Path:
    """
    Export a TIFF image to OME-Zarr format.
    
    Args:
        tiff_path: Path to input TIFF file
        output_dir: Path to output OME-Zarr directory
        pixel_size: Physical pixel size in micrometers (x, y)
        chunk_shape: Chunk dimensions (C, Y, X)
        coarsening_factor: Downsampling factor for pyramid levels
        max_levels: Maximum number of pyramid levels (None = auto)
        channel_labels: Optional list of channel names
        is_label_image: If True, treat as segmentation labels
        
    Returns:
        Path to created OME-Zarr directory
    """
    # Read TIFF
    image = imread(tiff_path)
    
    # Ensure CYX format
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    # Write OME-Zarr
    result_path = write_multiscale_omezarr(
        image=image,
        output_dir=output_dir,
        pixel_size=pixel_size,
        chunk_shape=chunk_shape,
        coarsening_factor=coarsening_factor,
        max_levels=max_levels,
    )
    
    # Add channel labels if provided
    if channel_labels is not None:
        import json
        attrs_path = result_path / ".zattrs"
        attrs = json.loads(attrs_path.read_text())
        for idx, label in enumerate(channel_labels):
            if idx < len(attrs["omero"]["channels"]):
                attrs["omero"]["channels"][idx]["label"] = label
        attrs_path.write_text(json.dumps(attrs, indent=2))
    
    # Add label-specific metadata if needed
    if is_label_image:
        import json
        attrs_path = result_path / ".zattrs"
        attrs = json.loads(attrs_path.read_text())
        attrs["image-label"] = {
            "version": "0.4",
            "source": {"image": "../../"},  # Relative path to source image
        }
        attrs_path.write_text(json.dumps(attrs, indent=2))
    
    return result_path
```

#### **Phase 4: Update Existing Modules to Use OME-Zarr**

**Goal**: Make downstream modules compatible with OME-Zarr inputs

**Changes Needed**:

1. **Update image reading utilities** in `lib/shared/image_utils.py`:
```python
def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Read image from TIFF or OME-Zarr format.
    
    Auto-detects format based on file extension/structure.
    """
    image_path = Path(image_path)
    
    if image_path.suffix in [".tif", ".tiff"]:
        from tifffile import imread
        return imread(image_path)
    
    elif image_path.suffix == ".zarr" or (image_path / ".zattrs").exists():
        import zarr
        store = zarr.DirectoryStore(str(image_path))
        group = zarr.open(store, mode="r")
        # Read highest resolution (scale0)
        return group["scale0"][:]
    
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
```

2. **Update configuration to support format selection**:
```yaml
all:
  image_format: omezarr  # or tiff (default)
  
preprocess:
  output_format: omezarr  # Export as OME-Zarr instead of TIFF
  
sbs:
  input_format: omezarr   # Read from OME-Zarr if available
  
phenotype:
  input_format: omezarr
```

3. **Create format-aware rule selection** in `workflow/Snakefile`:
```python
# Determine which conversion rule to use
if config.get("all", {}).get("image_format") == "omezarr":
    PREPROCESS_TARGETS_SBS = outputs_to_targets(
        {"convert_sbs_omezarr": PREPROCESS_OUTPUTS["convert_sbs_omezarr"]},
        sbs_wildcard_combos,
        PREPROCESS_OUTPUT_MAPPINGS,
    )
else:
    PREPROCESS_TARGETS_SBS = outputs_to_targets(
        {"convert_sbs": PREPROCESS_OUTPUTS["convert_sbs"]},
        sbs_wildcard_combos,
        PREPROCESS_OUTPUT_MAPPINGS,
    )
```

#### **Phase 5: Advanced Features**

**5.1 Parallel OME-Zarr Writing**

For large datasets, implement parallel chunk writing:

```python
def write_multiscale_omezarr_parallel(
    image: np.ndarray,
    output_dir: Path,
    n_workers: int = 4,
    **kwargs
) -> Path:
    """
    Parallel version of write_multiscale_omezarr.
    
    Uses multiprocessing to write chunks in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Build pyramid first
    pyramid = _build_pyramid(image, **kwargs)
    
    # Write levels in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for idx, level in enumerate(pyramid):
            future = executor.submit(
                _write_zarr_array,
                level,
                output_dir / f"scale{idx}",
                kwargs["chunk_shape"]
            )
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result()
    
    # Write metadata
    _write_group_metadata(output_dir, ...)
    
    return output_dir
```

**5.2 Compression Support**

Add compression to reduce storage size:

```python
def _write_zarr_array(
    level_data: np.ndarray,
    level_path: Path,
    chunk_shape: Sequence[int],
    compressor: str = "blosc",  # New parameter
) -> None:
    """Write with optional compression."""
    import numcodecs
    
    if compressor == "blosc":
        compressor_obj = numcodecs.Blosc(cname="zstd", clevel=5)
    elif compressor == "gzip":
        compressor_obj = numcodecs.GZip(level=6)
    else:
        compressor_obj = None
    
    # Update .zarray metadata
    zarray_meta = {
        "chunks": list(chunk_shape),
        "compressor": compressor_obj.get_config() if compressor_obj else None,
        # ... rest of metadata
    }
```

**5.3 Cloud Storage Support**

Enable writing to S3/GCS:

```python
def write_multiscale_omezarr(
    image: np.ndarray,
    output_dir: Union[str, Path],
    storage_backend: str = "local",  # or "s3", "gcs"
    storage_options: dict = None,
    **kwargs
) -> Path:
    """
    Write OME-Zarr to local or cloud storage.
    """
    if storage_backend == "s3":
        import s3fs
        fs = s3fs.S3FileSystem(**storage_options)
        store = s3fs.S3Map(root=str(output_dir), s3=fs)
    elif storage_backend == "gcs":
        import gcsfs
        fs = gcsfs.GCSFileSystem(**storage_options)
        store = gcsfs.GCSMap(root=str(output_dir), gcs=fs)
    else:
        store = Path(output_dir)
    
    # Write using zarr with appropriate store
    # ... implementation
```

### 5.5 Testing Strategy for OME-Zarr Exports

#### **Unit Tests**

Create `tests/unit/test_omezarr_export.py`:

```python
import pytest
import numpy as np
from pathlib import Path
from lib.shared.omezarr_utils import export_tiff_to_omezarr

def test_export_tiff_to_omezarr(tmp_path):
    """Test basic TIFF → OME-Zarr export."""
    # Create test TIFF
    from tifffile import imwrite
    test_image = np.random.randint(0, 65535, (3, 256, 256), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    imwrite(tiff_path, test_image)
    
    # Export to OME-Zarr
    zarr_path = tmp_path / "test.ome.zarr"
    result = export_tiff_to_omezarr(
        tiff_path=tiff_path,
        output_dir=zarr_path,
        channel_labels=["DAPI", "GFP", "RFP"],
    )
    
    # Verify structure
    assert (zarr_path / ".zattrs").exists()
    assert (zarr_path / "scale0" / ".zarray").exists()
    
    # Verify data
    import zarr
    z = zarr.open(str(zarr_path), mode="r")
    np.testing.assert_array_equal(z["scale0"][:], test_image)

def test_segmentation_export(tmp_path):
    """Test segmentation mask export with label metadata."""
    # Create test mask
    from tifffile import imwrite
    mask = np.random.randint(0, 100, (256, 256), dtype=np.uint16)
    tiff_path = tmp_path / "mask.tiff"
    imwrite(tiff_path, mask)
    
    # Export as label image
    zarr_path = tmp_path / "mask.ome.zarr"
    export_tiff_to_omezarr(
        tiff_path=tiff_path,
        output_dir=zarr_path,
        is_label_image=True,
    )
    
    # Verify label metadata
    import json
    attrs = json.loads((zarr_path / ".zattrs").read_text())
    assert "image-label" in attrs
```

#### **Integration Tests**

Extend `tests/integration/test_preprocess.py`:

```python
def test_convert_sbs_omezarr():
    """Test SBS OME-Zarr conversion."""
    omezarr_path = (
        PREPROCESS_FP
        / "omezarr"
        / "sbs"
        / get_filename(
            {"plate": TEST_PLATE, "well": TEST_WELL, "tile": TEST_TILE_SBS, "cycle": TEST_CYCLE},
            "image",
            "ome.zarr",
        )
    )
    
    # Verify structure
    assert (omezarr_path / ".zattrs").exists()
    assert (omezarr_path / ".zgroup").exists()
    assert (omezarr_path / "scale0").exists()
    
    # Verify metadata
    import json
    attrs = json.loads((omezarr_path / ".zattrs").read_text())
    assert "multiscales" in attrs
    assert len(attrs["multiscales"][0]["datasets"]) > 1  # Multiple scales
    
    # Verify data shape
    import zarr
    z = zarr.open(str(omezarr_path), mode="r")
    assert z["scale0"].shape == (5, 1200, 1200)
```

#### **End-to-End Tests**

```bash
# Test with OME-Zarr enabled
cd tests/small_test_analysis

# Modify config to enable OME-Zarr
sed -i 's/image_format: tiff/image_format: omezarr/' config/config.yml

# Run pipeline
snakemake --use-conda --cores all \
    --snakefile ../../workflow/Snakefile \
    --configfile config/config.yml \
    --until all_preprocess all_sbs all_phenotype

# Verify all OME-Zarr outputs exist
find brieflow_output/ -name "*.ome.zarr" -type d
```

### 5.6 Documentation Plan

Create comprehensive documentation for OME-Zarr workflows:

#### **User Guide** (`docs/omezarr_guide.md`):

```markdown
# OME-Zarr Usage Guide

## Enabling OME-Zarr Export

Add to your `config.yml`:

```yaml
all:
  image_format: omezarr  # Use OME-Zarr for all image outputs

preprocess:
  omezarr_chunk_shape: [1, 512, 512]
  omezarr_coarsening_factor: 2
  omezarr_max_levels: 5
```

## Visualizing OME-Zarr Images

### Using Napari

```python
import napari

viewer = napari.Viewer()
viewer.open("brieflow_output/preprocess/omezarr/sbs/P-1_W-A1_T-0_C-11__image.ome.zarr")
napari.run()
```

### Using Vizarr (web browser)

```bash
pip install vizarr
python -m vizarr brieflow_output/preprocess/omezarr/
```

### Using Python

```python
import zarr
import matplotlib.pyplot as plt

# Open OME-Zarr
z = zarr.open("path/to/image.ome.zarr", mode="r")

# Read highest resolution
img = z["scale0"][:]

# Plot first channel
plt.imshow(img[0], cmap="gray")
plt.show()
```

## Storage Considerations

OME-Zarr files are typically **larger** than TIFF due to pyramid levels,
but offer **better performance** for visualization and analysis.

Example sizes:
- TIFF: 100 MB
- OME-Zarr (5 levels): 133 MB (33% overhead)

Enable compression to reduce size:
```yaml
preprocess:
  omezarr_compression: blosc  # or gzip
```
```

#### **Developer Guide** (`docs/omezarr_development.md`):

```markdown
# OME-Zarr Development Guide

## Adding OME-Zarr Export to a New Module

1. Add target definition in `workflow/targets/<module>.smk`
2. Create export rule in `workflow/rules/<module>.smk`
3. Create export script in `workflow/scripts/<module>/`
4. Add tests in `tests/integration/`

## OME-Zarr Specification

Brieflow follows OME-NGFF v0.4 specification:
https://ngff.openmicroscopy.org/0.4/

## Metadata Structure

```json
{
  "multiscales": [...],  // Pyramid metadata
  "omero": {             // Visualization metadata
    "channels": [...],
    "pixel_size": {...}
  },
  "image-label": {...}   // For segmentation masks only
}
```
```

### 5.7 Performance Benchmarks

Create benchmarking script `scripts/benchmark_omezarr.py`:

```python
"""Benchmark OME-Zarr write performance."""

import time
import numpy as np
from pathlib import Path
from tifffile import imwrite
from lib.preprocess.preprocess import write_multiscale_omezarr

def benchmark_write_performance():
    """Compare TIFF vs OME-Zarr write times."""
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    channels = [3, 5, 10]
    
    results = []
    
    for size in sizes:
        for n_channels in channels:
            # Generate test image
            image = np.random.randint(0, 65535, (n_channels, *size), dtype=np.uint16)
            
            # Benchmark TIFF
            tiff_path = Path(f"/tmp/test_{size[0]}_{n_channels}.tiff")
            start = time.time()
            imwrite(tiff_path, image)
            tiff_time = time.time() - start
            tiff_size = tiff_path.stat().st_size / 1024 / 1024  # MB
            
            # Benchmark OME-Zarr
            zarr_path = Path(f"/tmp/test_{size[0]}_{n_channels}.ome.zarr")
            start = time.time()
            write_multiscale_omezarr(
                image=image,
                output_dir=zarr_path,
                chunk_shape=(1, 512, 512),
                coarsening_factor=2,
                max_levels=5,
            )
            zarr_time = time.time() - start
            
            # Calculate total size of OME-Zarr directory
            zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()) / 1024 / 1024
            
            results.append({
                "size": size[0],
                "channels": n_channels,
                "tiff_time": tiff_time,
                "zarr_time": zarr_time,
                "tiff_size": tiff_size,
                "zarr_size": zarr_size,
                "zarr_overhead": (zarr_size / tiff_size - 1) * 100,
            })
    
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    benchmark_write_performance()
```

Expected output:
```
 size  channels  tiff_time  zarr_time  tiff_size  zarr_size  zarr_overhead
  512         3       0.05       0.15       1.50       1.80          20.0
  512         5       0.08       0.22       2.50       3.00          20.0
 1024         3       0.15       0.45       6.00       7.50          25.0
 1024         5       0.25       0.70      10.00      12.50          25.0
 2048         3       0.60       1.80      24.00      30.00          25.0
 2048         5       1.00       2.80      40.00      50.00          25.0
```

---

## 6. Implementation Roadmap

### 6.1 Timeline

#### **Week 1-2: Testing & Validation**
- [x] Set up test environment
- [ ] Run complete pipeline on small_test_analysis
- [ ] Document any issues encountered
- [ ] Validate all module outputs
- [ ] Create baseline metrics

#### **Week 3-4: OME-Zarr Infrastructure**
- [ ] Review existing OME-Zarr code
- [ ] Create `lib/shared/omezarr_utils.py`
- [ ] Write unit tests for new utilities
- [ ] Benchmark write performance
- [ ] Document API

#### **Week 5-6: SBS Module OME-Zarr Export**
- [ ] Add OME-Zarr targets for SBS outputs
- [ ] Create export rules and scripts
- [ ] Test on small_test_analysis
- [ ] Verify visualization in Napari
- [ ] Document usage

#### **Week 7-8: Phenotype Module OME-Zarr Export**
- [ ] Add OME-Zarr targets for phenotype outputs
- [ ] Implement segmentation mask export
- [ ] Create export rules and scripts
- [ ] Test on small_test_analysis
- [ ] Document usage

#### **Week 9-10: Integration & Optimization**
- [ ] Make downstream modules compatible with OME-Zarr inputs
- [ ] Implement parallel writing
- [ ] Add compression support
- [ ] End-to-end testing
- [ ] Performance benchmarking

#### **Week 11-12: Documentation & Release**
- [ ] Write user guide
- [ ] Write developer guide
- [ ] Create tutorial notebooks
- [ ] Update README.md
- [ ] Prepare release notes

### 6.2 Success Criteria

#### **Minimum Viable Product (MVP)**:
✅ OME-Zarr export works for all preprocess outputs  
✅ OME-Zarr export works for aligned SBS images  
✅ OME-Zarr export works for aligned phenotype images  
✅ Downstream modules can read OME-Zarr inputs  
✅ Tests pass with OME-Zarr enabled  
✅ Basic documentation available  

#### **Full Feature Set**:
✅ All above criteria  
✅ Segmentation mask export as OME-Zarr labels  
✅ Multi-resolution pyramids (3-5 levels)  
✅ Parallel chunk writing  
✅ Compression support  
✅ Cloud storage support (S3/GCS)  
✅ Comprehensive documentation  
✅ Performance benchmarks  

### 6.3 Risk Mitigation

#### **Risk: OME-Zarr files are too large**
- **Mitigation**: Implement compression (Blosc/gzip)
- **Mitigation**: Make pyramid levels configurable
- **Mitigation**: Provide option to export only specific stages

#### **Risk: Downstream tools don't support OME-Zarr**
- **Mitigation**: Keep TIFF export as default option
- **Mitigation**: Provide conversion utilities
- **Mitigation**: Update image reading utilities to support both formats

#### **Risk: Write performance is too slow**
- **Mitigation**: Implement parallel writing
- **Mitigation**: Optimize chunk sizes
- **Mitigation**: Profile and optimize bottlenecks

#### **Risk: Compatibility issues with visualization tools**
- **Mitigation**: Test with Napari, Vizarr, Neuroglancer
- **Mitigation**: Strictly follow OME-NGFF v0.4 spec
- **Mitigation**: Validate metadata with ome-zarr-py validator

### 6.4 Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| SBS aligned image export | High | Low | **P0** |
| Phenotype aligned image export | High | Low | **P0** |
| Segmentation mask export | High | Medium | **P1** |
| OME-Zarr input support | High | Medium | **P1** |
| Compression support | Medium | Low | **P2** |
| Parallel writing | Medium | Medium | **P2** |
| Cloud storage support | Low | High | **P3** |
| Advanced visualization | Low | Medium | **P3** |

---

## 7. Key Takeaways

### 7.1 Brieflow Architecture Summary

1. **Modular Design**: Six independent modules (preprocess, sbs, phenotype, merge, aggregate, cluster)
2. **Snakemake-Based**: Declarative workflow with automatic dependency resolution
3. **Library-First**: Core logic in `workflow/lib/`, scripts are thin wrappers
4. **Flexible Configuration**: YAML-based configuration with per-module parameters
5. **Production-Ready**: Existing test suite, HPC integration, documentation

### 7.2 Testing Approach

1. **Start Small**: Use `small_test_analysis` (14 min runtime)
2. **Module-by-Module**: Test each module independently before integration
3. **Validation First**: Run existing tests before making changes
4. **Continuous Testing**: Re-test after each modification

### 7.3 OME-Zarr Implementation Strategy

1. **Leverage Existing Code**: OME-Zarr writing already implemented in preprocess module
2. **Incremental Rollout**: Start with SBS, then phenotype, then other modules
3. **Backwards Compatible**: Keep TIFF export as default, make OME-Zarr optional
4. **Test Extensively**: Unit tests, integration tests, end-to-end tests
5. **Document Thoroughly**: User guide, developer guide, tutorials

### 7.4 Next Immediate Actions

1. **Run complete test pipeline** to establish baseline
2. **Review existing OME-Zarr code** to understand current implementation
3. **Create `omezarr_utils.py`** library module for reusable functions
4. **Implement SBS aligned export** as first OME-Zarr extension
5. **Test and iterate** before expanding to other modules

---

## Appendix A: Key File Locations

### Critical Library Files
- `workflow/lib/preprocess/preprocess.py` - OME-Zarr writer implementation
- `workflow/lib/shared/image_utils.py` - Image processing utilities
- `workflow/lib/shared/target_utils.py` - Snakemake target management
- `workflow/lib/sbs/call_reads.py` - Read calling algorithms
- `workflow/lib/phenotype/extract_phenotype_cp_multichannel.py` - Feature extraction

### Critical Rule Files
- `workflow/rules/preprocess.smk` - Lines 97-116 (convert_sbs_omezarr), 137-156 (convert_phenotype_omezarr)
- `workflow/rules/sbs.smk` - SBS pipeline rules
- `workflow/rules/phenotype.smk` - Phenotype pipeline rules

### Critical Test Files
- `tests/integration/test_omezarr_writer.py` - OME-Zarr unit tests
- `tests/integration/test_preprocess.py` - Preprocess integration tests
- `tests/small_test_analysis/run_brieflow.sh` - End-to-end test script

### Configuration Files
- `tests/small_test_analysis/config/config.yml` - Example configuration
- `pyproject.toml` - Package dependencies

---

## Appendix B: Glossary

- **SBS**: Sequencing by Synthesis - in situ sequencing technique
- **OPS**: Optical Pooled Screening - high-throughput imaging method
- **OME-Zarr**: Open Microscopy Environment Zarr format
- **OME-NGFF**: OME Next-Generation File Format specification
- **CP**: CellProfiler - image analysis software
- **IC**: Illumination Correction
- **PHATE**: Potential of Heat-diffusion for Affinity-based Trajectory Embedding
- **Leiden**: Community detection algorithm for clustering

---

**Document Version**: 1.0  
**Last Updated**: December 2, 2025  
**Author**: AI Assistant  
**Review Status**: Draft

