# Glossary

Terms used across these docs, the notebooks and the pipeline outputs.
Notebook parameters (the UPPERCASE names) have their own page: [Notebook Parameters](notebook_parameters.md).

## Screens and imaging

```{glossary}
OPS
Optical pooled screen
  A genetic screen in which a pooled library of perturbations is introduced into cells, the cells are imaged for a phenotype, and each cell's perturbation is then read out in place by in situ sequencing of a barcode. Brieflow links the two readouts cell by cell.

Plate
  One physical multi-well plate. Brieflow's `plate` wildcard is an integer, and plates are processed one at a time in preprocess, SBS and phenotype by default.

Well
  One well of a plate, written as a row letter and column number (`A1`) or, for Opera Phenix data, `r02c02`. In zarr mode a well becomes a row/column group in the plate store (`A/1`).

Tile
  One field of view within a well, numbered from 0. SBS and phenotype are imaged at different magnifications, so their tiles do not line up and merge has to match them.

Site
  An SBS tile, in merge. The merge code pairs a phenotype `tile` with an SBS `site` when it aligns the two imaging runs.

Cycle
  One round of in situ sequencing chemistry, which reads one base of every barcode. An SBS experiment with N cycles reads the first N bases.

SBS
Sequencing by synthesis
  The in situ sequencing arm of a screen: the imaging cycles that read each cell's barcode, and the brieflow module that aligns those cycles, finds spots, calls reads and assigns barcodes to cells.

Phenotype
  The imaging arm that measures what the perturbation did (stains, antibodies, fluorescent tags), and the brieflow module that aligns, segments and extracts features from those images.

IC field
Illumination correction field
  A per-channel image of uneven illumination, computed by preprocess from many tiles of a plate (after CellProfiler's CorrectIlluminationCalculate) and divided out of every tile before analysis. `SAMPLE_FRACTION` below 1 computes it from a random subset of tiles drawn with a fixed seed.
```

## Segmentation and features

```{glossary}
Segmentation
  Finding the pixels that belong to each nucleus and each cell. Brieflow segments SBS and phenotype images separately, with Cellpose, StarDist or, for SBS only, a threshold-based watershed.

Masks
Nuclei mask
Cell mask
Cytoplasm mask
  Label images in which every pixel of an object carries that object's integer ID. Segmentation produces nuclei and cell masks; the cytoplasm mask is each cell mask with its nucleus removed. In zarr mode, masks are stored as labels inside the image store.

Reconcile
  How nuclei and cell masks are paired after segmentation. `contained_in_cells` (the notebook default) keeps cells containing more than one nucleus, merging their nuclei, which suits dividing cells; `consensus` keeps only nucleus/cell pairs whose match is one-to-one.

Cellpose
cpsam
  [Cellpose](https://www.cellpose.org) is a deep-learning segmentation model. brieflow installs Cellpose 3 (models such as `cyto3`, the default); `cpsam` is the Cellpose-SAM model of Cellpose 4, installed separately, and slow without a GPU.

Secondary object
  An object segmented inside cells in addition to nuclei and cells, such as an intracellular pathogen, an organelle or foci, switched on by `SECOND_OBJ_DETECTION` in the phenotype notebook. Each object gets its own features and is assigned to the cell that contains it.

CP emulator
cp_measure
  The two feature extractors in the phenotype module, chosen by `CP_METHOD`. `cp_emulator` (the default) reimplements CellProfiler-style measurements following Feldman et al. 2019; `cp_measure` calls the [cp_measure](https://github.com/afermg/cp_measure) package, a Python port of CellProfiler's measurements.

Compartment
  The region a feature is measured in: `nucleus`, `cell`, `cytoplasm`, or `second_obj` for secondary objects. Feature names start with it, for example `nucleus_DAPI_mean`.
```

## Barcodes and perturbations

```{glossary}
Barcode
  The sequence read by SBS that identifies a cell's perturbation. In a CRISPR screen it is usually the sgRNA itself (or its first N bases, one per cycle); multi-barcode designs read two regions (MAP and RECOMB).

Guide
sgRNA
  The single guide RNA that directs Cas9 to its target gene. One gene is usually targeted by several guides.

Construct
  One distinct perturbation reagent, identified by its barcode or sgRNA (`PERTURBATION_ID_COL`, for example `cell_barcode_0`). Aggregate and the bootstrap work per construct before rolling up to genes, so each non-targeting guide is its own control element.

Perturbation
  What a cell's construct targets, named by `PERTURBATION_NAME_COL` (usually the gene symbol). Aggregate produces one profile per perturbation.

Mapping rate
  How much of the SBS signal matches the barcode library. The read mapping rate is the fraction of reads above the read threshold whose sequence is a library barcode; the cell mapping rate is the fraction of cells whose barcode maps to one library entry. The SBS notebook and `sbs/eval/` plot both.

Merge
  The module that assigns each phenotype cell its SBS cell, and so its barcode. The default `fast` approach aligns each phenotype tile to SBS sites by hashing triangles of nuclear centroids, refines the fit with a RANSAC regression seeded at 0, matches cells within a distance threshold and removes duplicate matches; the `stitch` approach stitches whole wells first.
```

## Aggregation and statistics

```{glossary}
Channel combo
  A set of phenotype channels whose features are aggregated and clustered together, named by joining the channels with `_` (for example `DAPI_COXIV_CENPA_WGA`). Set in the aggregate notebook as `CHANNEL_COMBOS`; each gets its own output directory.

Compartment combo
  A set of compartments whose features are aggregated together when `SPLIT_BY_COMPARTMENT` is on, named by joining them with `-` (for example `cell-nucleus-cytoplasm`). Every channel combo is paired with every compartment combo.

Cell class
  One dataset that aggregate and cluster process separately, such as `Interphase` or `Mitotic` from a classifier, or one value of an annotation column. An `all` class containing every cell is always made.

Per-well annotations
SPLIT_COL
  Per-well annotations are a TSV (`WELL_ANNOTATIONS_FP`) with columns `plate`, `well` and one column per experimental variable, joined onto every cell in aggregate; wells missing from it are left out. `SPLIT_COL` names the column whose values define the cell classes: `class` (the classifier output, the default) or an annotation column.

Aggregate
  The module that turns merged single cells into per-perturbation profiles: filtering, missing-value handling, perturbation scoring, PCA and TVN alignment, then a median or mean per construct and per gene, plus optional bootstrap statistics.

Perturbation score
  A per-cell score of how strongly a cell shows its perturbation's phenotype, from a logistic regression separating that perturbation's cells from controls. Aggregate can keep only cells above a probability or percentile threshold, and the regression's AUC per perturbation (`perturbation_auc`) lets cluster drop weak perturbations with `PERTURBATION_AUC_THRESHOLD`.

TVN
Typical variation normalization
Batch correction
  Normalization that centers and scales the PCA embedding on control cells and then whitens it with CORAL so control variation is equal in every direction. With `TVN_BATCH_CORRECTION` on (the default) it is fit per batch (`BATCH_COLS`, usually plate and well) on that batch's controls, which also corrects batch effects; off, it is fit once on pooled controls.

Control key
  The perturbation value that marks control cells, for example `nontargeting` (`CONTROL_KEY`, `aggregate.control_key`). Cluster can override it with `CLUSTER_CONTROL_KEY` (`cluster.control_key`).

Control scope
  Which controls form the null that a point is compared with, in the aggregate bootstrap (`BOOTSTRAP_CONTROL_SCOPE`) and in cluster's distance to controls (`CONTROL_SCOPE`). `pooled` uses all controls; `within_group` uses controls in the point's own `GROUP_COLS` group; `reference_group` uses controls in the named reference group, such as the vehicle; `within_perturbation` uses the point's own perturbation in the reference group, so each treated arm is scored against its own vehicle arm and the control key plays no part.

Bootstrap scoring
  Statistical testing of each construct and gene against a null built by repeatedly sampling control cells: each simulation draws one control construct and takes the median of a sample of its cells, and a feature's two-tailed p-value is how often the null medians are more extreme than the observed one, followed by FDR correction.
```

## Clustering and annotation

```{glossary}
PHATE
  A dimensionality reduction that preserves local and global structure, used by cluster to embed perturbation profiles in two dimensions.

Leiden clustering
Leiden resolution
  Graph-based community detection on the PHATE diffusion graph. The resolution sets granularity: higher values give more, smaller clusters. Cluster writes one output directory per resolution in `FINAL_LEIDEN_RESOLUTIONS`.

Benchmarks
  Reference gene sets used to score clusterings: CORUM protein complexes, KEGG pathways (group benchmarks) and STRING interactions (pair benchmarks) for the organism set by `SPECIES_ID`. Each resolution is scored against the real benchmark and a shuffled one.

MozzareLLM
  A [package](https://github.com/cheeseman-lab/mozzarellm) that asks a language model to name the process behind each gene cluster and to classify each gene in it, run by `flow.sh mozzarellm` on the clusterings chosen in `12_analyze.py`.
```

## Configuration and running

```{glossary}
screen.yaml
  The description of the screen in `analysis/screen.yaml`: organism and cell line, library, SBS and phenotype channels and stains, microscope, and where the raw data lives. The pipeline does not read it; `12_analyze.py` builds MozzareLLM screen contexts from it, the visualizer shows it, and brieflow-auto derives notebook parameters from it.

config.yml
  `analysis/config/config.yml`, the single Snakemake config for a screen. Each notebook writes its own section (`all`, `preprocess`, `sbs`, `phenotype`, `merge`, `classify`, `aggregate`, `cluster`, `mozzarellm`); `flow.sh` passes the file to Snakemake.

Marker block
Operator parameters
  The cells of a notebook between `# === OPERATOR PARAMETERS ===` and `# === END OPERATOR PARAMETERS ===`, holding the UPPERCASE values a person (or brieflow-auto) sets. Everything else in the notebook is derived from them.

flow.sh module
  A name passed to `analysis/flow.sh` that runs one part of the pipeline: `preprocess`, `sbs`, `phenotype`, `merge`, `aggregate`, `cluster`, `mozzarellm`, `viz`, or `all` for preprocess through cluster.

TIFF mode
Zarr mode
  The two output layouts, chosen by `all.image_format`. TIFF mode writes one file per tile and flat file names; zarr mode writes OME-Zarr plate stores for images and nests per-tile tables in `plate/row/col/tile` directories. Pipeline parameters are the same in both.

OME-Zarr
  A chunked, cloud-friendly image format from the OME community. Brieflow writes one store per plate and image type, laid out as a high-content-screening plate (row, column, field), with channel metadata from the preprocess notebook.
```

## brieflow-auto

```{glossary}
Interview
  brieflow-auto's resolution of every notebook operator parameter to a value and a source (`screen.yaml`, a lab default, a data probe, an agent-tuned sweep, or a question to the operator), recorded in `.brieflow/interview.json`.

Launch review
  The page brieflow-auto writes to `.brieflow/launch_review.html` before a run: the phases in range, every active parameter with its provenance, gaps, disk needs and compute target. The run starts only after the operator re-runs with `--approved`.

Run manifest
  `.brieflow/run_manifest.json`, written at every launch: plugin, brieflow and screen repository versions, interpreter and library versions, and the flags used. It identifies a run; it does not guarantee a rerun reproduces it.

QC gate
  brieflow-auto's pass/fail check on a finished phase, reading the phase's `eval/` outputs against thresholds in its `qc_gates.yaml`. A failed gate pauses the run.
```
