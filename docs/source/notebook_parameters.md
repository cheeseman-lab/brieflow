# Notebook Parameters

The marimo notebooks hold a few hundred settable values, the UPPERCASE variables in their {term}`operator parameter <Marker block>` cells.
This page covers the ones a researcher has to understand or commonly sets, grouped by notebook.
Each entry gives the config key it writes (or what it feeds, when it is not written directly), its default in the notebook, and when to change it.
The notebook's own **SET PARAMETERS** cells remain the full reference.

A default of `None` means there is no usable default: the value is specific to your screen and must be set.

## Shared by several notebooks

```{glossary}
CONFIG_FILE_PATH
  Every notebook · default `"config/config.yml"`. The config file each notebook reads and rewrites. Leave it unless you keep several configs.

TEST_PLATE, TEST_WELL, TEST_TILE
  SBS and phenotype notebooks (merge uses `TEST_PLATE` and `TEST_WELL`; aggregate uses `TEST_PLATE`, `TEST_WELL_1` and `TEST_WELL_2`) · not written · default `None`. The data the notebook loads to test its settings. Pick a tile from the middle of a well with typical cell density; edge tiles are often partly empty.
```

## Preprocess (`0_preprocess.py`)

```{glossary}
ROOT_FP
  `all.root_fp` · default `"brieflow_output/"`. Root directory of all pipeline outputs, relative to `analysis/` or absolute. Point it at larger storage for a big screen.

IMAGE_FORMAT
  `all.image_format` · default `"zarr"`. Output layout for the whole run: `"zarr"` for {term}`OME-Zarr` stores, `"tiff"` for one TIFF per tile. Choose once, before preprocess runs; switching later means re-running from preprocess.

SBS_IMAGES_DIR_FP, PHENOTYPE_IMAGES_DIR_FP
  Not written; they feed the sample tables (`preprocess.sbs_samples_fp`, `preprocess.phenotype_samples_fp`) · default `None`. Directories holding the raw SBS and phenotype files. Leave one as `None` to process only the other arm.

SBS_PATH_PATTERN, PHENOTYPE_PATH_PATTERN
  Not written; they build the sample tables · default `None`. Regular expressions matched against each file path, with one capture group per entry of `SBS_PATH_METADATA` / `PHENOTYPE_PATH_METADATA` (at least plate, well and tile, plus cycle for SBS). The notebook shows the table they produce; check it lists every file.

SBS_DATA_FORMAT, PHENOTYPE_DATA_FORMAT
  `preprocess.sbs_data_format`, `preprocess.phenotype_data_format` · default `None`. The raw file type: `"nd2"`, `"ims"` or `"tiff"`. TIFF inputs usually also need the external metadata settings (`*_METADATA_*`) for stage positions.

SBS_DATA_ORGANIZATION, PHENOTYPE_DATA_ORGANIZATION
  `preprocess.sbs_data_organization`, `preprocess.phenotype_data_organization` · default `None`. `"tile"` when each file holds one field of view, `"well"` when a file holds a whole well. A single-position well file is read as one tile.

SBS_CHANNEL_ORDER, PHENOTYPE_CHANNEL_ORDER
  `preprocess.sbs_channel_order`, `preprocess.phenotype_channel_order` · default `None`. The channel order to stack, when each file holds a single channel (or a tile spans several files). Leave `None` for multichannel files. DAPI must end up first.

PHENOTYPE_ROUND_ORDER
  `preprocess.phenotype_round_order` · default `None`. The order in which to stack phenotype rounds when a tile was imaged in several rounds, for example `[1, 2]`. `None` for a single round.

SBS_CHANNEL_ORDER_FLIP, PHENOTYPE_CHANNEL_ORDER_FLIP
  `preprocess.sbs_channel_order_flip`, `preprocess.phenotype_channel_order_flip` · default `False`. Reverse the channel order of multichannel files on conversion. Set `True` when the test conversion shows DAPI last.

SBS_N_Z_PLANES, PHENOTYPE_N_Z_PLANES
  `preprocess.sbs_n_z_planes`, `preprocess.phenotype_n_z_planes` · default `None`. Number of z-planes per channel when TIFF files are split by plane; the planes are stacked and max-projected. Requires `"z"` in the path metadata and an explicit channel order.

SAMPLE_FRACTION
  `preprocess.sample_fraction` · default `1.0`. Fraction of tiles used to compute each {term}`IC field`. Lower it (for example `0.2`) to cut memory and time on large plates; the subset is drawn with a fixed seed.

SBS_CHANNELS_METADATA, PHENOTYPE_CHANNELS_METADATA
  `preprocess.sbs_channels_metadata`, `preprocess.phenotype_channels_metadata` · default `None`. One dictionary per channel (name, index, type, description, optional biological annotation), written into the OME-Zarr plate metadata. Same length and order as the channel names.
```

## SBS (`2_sbs.py`)

```{glossary}
CHANNEL_NAMES (SBS)
  `sbs.channel_names` · default `None`. Names of the SBS channels in image order, for example `["DAPI", "G", "T", "A", "C"]`.

BASES, EXTRA_CHANNELS
  `BASES` writes `sbs.bases`; `EXTRA_CHANNELS` feeds `sbs.extra_channel_indices` · default `None`. Leave both `None` for four-color G/T/A/C imaging. For combinatorial chemistry, `BASES` names the dye channels and `EXTRA_CHANNELS` the channels that encode no base, such as DAPI.

ALIGNMENT_METHOD
  `sbs.alignment_method` · default `None` (chosen automatically). How cycles are aligned: `"DAPI"` when every cycle has DAPI, otherwise `"sbs_mean"` (mean of the base channels).

SKIP_CYCLES
  Feeds `sbs.skip_cycles_indices` · default `None`. Cycle numbers to leave out, for example a failed cycle.

SPOT_DETECTION_METHOD
  `sbs.spot_detection_method` · default `"standard"`. `"standard"` finds spots from the standard deviation across cycles; `"spotiflow"` uses the Spotiflow deep-learning detector on one cycle.

DAPI_CYCLE, CYTO_CYCLE, CYTO_CHANNEL (SBS)
  `sbs.dapi_cycle`, `sbs.cyto_cycle`; `CYTO_CHANNEL` feeds `sbs.cyto_index` · default `None`. Which cycle holds DAPI, and which cycle and channel mark cell boundaries for SBS segmentation. With a cellular stain, set `CYTO_CYCLE` equal to `DAPI_CYCLE`.

SEGMENTATION_METHOD (SBS)
  `sbs.segmentation_method` · default `"cellpose"`. `"cellpose"`, `"stardist"` or `"watershed"`.

SEGMENT_CELLS (SBS)
  `sbs.segment_cells` · default `True`. Segment cell bodies as well as nuclei. Set `False` when barcode spots are nuclear (DNA barcodes), which is faster.

CELLPOSE_MODEL
  `sbs.cellpose_model` or `phenotype.cellpose_model` · default `"cyto3"`. Cellpose model: `"cyto3"`, `"cyto2"`, `"cyto"`, `"nuclei"`, or `"cpsam"`, which needs Cellpose 4 and ideally a GPU.

GPU (SBS)
  `sbs.gpu` · default `False`. Run SBS segmentation on a GPU, in the notebook and in the pipeline.

BARCODE_TYPE
  `sbs.barcode_type` · default `"simple"`. `"simple"` for one barcode per construct; `"multi"` for designs with MAP and RECOMB regions, which have their own column and cycle-range settings.

DF_DESIGN_FP, BARCODE_COL
  `DF_DESIGN_FP` is not written; the notebook standardizes it into the library at `sbs.df_barcode_library_fp`. `BARCODE_COL` writes `sbs.barcode_col` · default `None`. The raw guide design table and the column holding each barcode sequence, with `GENE_SYMBOL_COL` and `GENE_ID_COL` naming the gene columns.

SPECIES_ID (SBS)
  Not written · default `"9606"` (human). NCBI taxonomy ID used to fetch UniProt annotations for the library genes.

CHEMISTRY
  `sbs.chemistry` · default `"four_color"`. `"four_color"` when each base has its own channel; `"combinatorial"` when a base is an ON/OFF pattern across dye channels, set in `COMBINATORIAL_CODE` (`sbs.combinatorial`).

CALL_READS_METHOD
  `sbs.call_reads_method` · default `"median"`. How intensities become reads: `"median"` or `"percentile"` for four-color; `"frac"` (recommended) or `"merfish"` for combinatorial.

THRESHOLD_READS
  `sbs.threshold_peaks` · default `50`. Minimum peak intensity for a spot to become a read. Tune it with the notebook's {term}`mapping rate` curve: higher is stricter.

Q_MIN
  `sbs.q_min` · default `0`. Minimum base quality score for a read to be kept.

SORT_CALLS
  `sbs.sort_calls` · default `"count"`. How a cell's barcodes are ranked among its reads: `"count"` (by read frequency, for mRNA protocols) or `"peak"` (by intensity, for DNA protocols).

ERROR_CORRECT, MAX_DISTANCE
  `sbs.error_correct`, `sbs.max_distance` · defaults `False`, `None`. Allow a called barcode to be corrected to a library barcode up to `MAX_DISTANCE` bases away. Can be used with combinatorial `frac` (start at `1`); must stay `False` with `merfish`.
```

## Phenotype (`3_phenotype.py`)

```{glossary}
CHANNEL_NAMES (phenotype)
  `phenotype.channel_names` · default `None`. Names of the phenotype channels in image order. These become feature names (`nucleus_DAPI_mean`, …).

ALIGN
  `phenotype.align` · default `None`. Align phenotype channels to each other; needed unless all channels were captured consecutively. `TARGET` and `SOURCE` name the channels to align (written as indices), `RIDERS` follow the source, and `REMOVE_CHANNEL` drops a duplicate channel used only for alignment.

CYTO_CHANNEL (phenotype)
  Feeds `phenotype.cyto_index` · default `None`. The channel that marks cell boundaries for cell segmentation.

SEGMENTATION_METHOD (phenotype)
  `phenotype.segmentation_method` · default `"cellpose"`. `"cellpose"` or `"stardist"`.

SEGMENT_CELLS (phenotype)
  `phenotype.segment_cells` · default `True`. Segment cells as well as nuclei. `False` gives nuclear features only, faster.

RECONCILE
  `phenotype.reconcile` (also `sbs.reconcile`) · default `"contained_in_cells"`. How nuclei and cells are paired; see {term}`Reconcile`.

GPU (phenotype)
  Not written directly · default `False`. Use a GPU for the notebook's test segmentations. It also becomes `phenotype.gpu` unless `PIPELINE_GPU` is set.

PIPELINE_GPU
  `phenotype.gpu` · default `None` (use `GPU`). GPU setting for the pipeline's `segment_phenotype` and `identify_second_objs` jobs. Set `False` to tune on a GPU node but segment on CPU, or `True` for the reverse.

CP_METHOD
  `phenotype.cp_method` · default `"cp_emulator"`. Feature extractor: `"cp_emulator"` or `"cp_measure"`; see {term}`CP emulator`.

FOCI_CHANNEL
  Feeds `phenotype.foci_channel_index` · default `None`. Channel (or list of channels) in which to measure foci, for example `"GH2AX"`. `None` skips foci features.

CUSTOM_FEATURES
  `phenotype.custom_features` (the function source) · default `[]`. Extra per-cell measurement functions, each measured on the nucleus or a declared compartment. Requires `CP_METHOD = "cp_emulator"`.

SECOND_OBJ_DETECTION
  `phenotype.second_obj_detection` · default `False`. Turn on {term}`secondary object <Secondary object>` detection.

SECOND_OBJ_CHANNEL
  Feeds `phenotype.second_obj_channel_index` · default `None`. The channel carrying the secondary objects.

SECOND_OBJ_METHOD
  `phenotype.second_obj_method` · default `"threshold"`. `"threshold"`, `"cellpose"` or `"stardist"`, each with its own size filter and method settings in the notebook.
```

## Merge (`5_merge.py`)

```{glossary}
INITIAL_SITES_APPROACH
  Not written · default `None`. How the starting tile pairs are given: `"auto"` finds a phenotype tile for each of `INITIAL_SBS_TILES` from stage coordinates; `"manual"` takes explicit `INITIAL_SITES`.

INITIAL_SITES, INITIAL_SBS_TILES
  `merge.initial_sites`, `merge.initial_sbs_tiles` · default `None`. Starting pairs `[phenotype_tile, sbs_tile]`, or SBS tiles spread across the well. At least 5 pairs should pass `DET_RANGE` and the score threshold.

DET_RANGE
  `merge.det_range` · default `None`. The accepted range of the alignment's scale (determinant), which reflects the magnification ratio between phenotype and SBS images, for example `[0.06, 0.065]`.

THRESHOLD
  `merge.threshold` · default `None`. Maximum distance between a phenotype cell and an SBS cell for them to match, for example `2`.

STITCH
  Writes `merge.approach` (`"stitch"` if `True`, else `"fast"`) · default `False`. Use the stitch approach when no good initial sites can be found.
```

## Classify (`7_classify.py`)

```{glossary}
TRAINING_DATA_SOURCE
  Not written · default `"merge"`. Which features the classifier is trained on: `"merge"` or `"phenotype"` output.

CLASS_TITLE
  `classify.class_title` · default `"cell_stage"`. Name of the column holding the predicted class.

CLASSIFICATION
  Feeds `classify.class_mapping` · default `["Mitotic", "Interphase"]`. The classes to label and predict, numbered 1, 2, … in list order.

CONFIDENCE_THRESHOLDS
  `classify.confidence_thresholds`. Per-class confidence thresholds keyed by class number, each `{"threshold": float, "mode": str}`; the mode (`"exclude"` or `"reassign"`) sets how a prediction below the threshold is handled.
```

## Aggregate (`8_aggregate.py`)

```{glossary}
WELL_ANNOTATIONS_FP
  `aggregate.well_annotations_fp` · default `None`. TSV of {term}`per-well annotations <Per-well annotations>` (`plate`, `well`, then one column per variable). Only annotated wells are aggregated.

SPLIT_COL (parameter)
  `aggregate.split_col` · default `"class"`. Column whose values become separate datasets ({term}`cell classes <Cell class>`), each with its own embedding. `"class"` uses the classifier; an annotation column (for example `"treatment"`) splits by its values.

GROUP_COLS
  `aggregate.group_cols` · default `[]`. Annotation columns to aggregate within, so a profile is perturbation × group in one shared embedding, for example `["treatment"]`.

BOOTSTRAP_CONTROL_SCOPE, BOOTSTRAP_REFERENCE_GROUP
  `aggregate.bootstrap_control_scope`, `aggregate.bootstrap_reference_group` · defaults `"pooled"`, `None`. The {term}`control scope <Control scope>` of the bootstrap null, and the group (usually the vehicle) that `reference_group` and `within_perturbation` pin it to. Change from `pooled` when `GROUP_COLS` groups have different baselines or are treatments.

CHANNEL_COMBOS
  Written to the aggregate combo table (`aggregate.aggregate_combo_fp`) · default `None`. The {term}`channel combos <Channel combo>` to aggregate, as lists of channel names.

SPLIT_BY_COMPARTMENT, COMPARTMENT_COMBOS
  `aggregate.split_by_compartment`; combos go to the combo table · defaults `False`, `None`. Aggregate and cluster separately per {term}`compartment combo <Compartment combo>`.

SECOND_OBJ_AGG_STRATEGY
  `aggregate.second_obj_agg_strategy` · default `"none"`. How per-object features fold into the cell: `"none"`, `"single"` (cells with exactly one object), `"all"` (numbered columns per object) or `"average"`.

FILTER_QUERIES
  `aggregate.filter_queries` · default `None`. pandas query strings that cells must pass, for example `["mapped_single_gene == True"]`.

PERTURBATION_NAME_COL
  `aggregate.perturbation_name_col` · default `None`. Column that names each cell's {term}`perturbation <Perturbation>` and that profiles are aggregated on, usually `"gene_symbol_0"`.

PERTURBATION_ID_COL
  `aggregate.perturbation_id_col` · default `"cell_barcode_0"`. Column identifying each {term}`construct <Construct>`. Should essentially always be set; `None` collapses constructs to genes and makes bootstrap p-values under-dispersed.

CONTROL_KEY
  `aggregate.control_key` · default `None`. The {term}`control key <Control key>`, for example `"nontargeting"`. `CONTROL_NAME_COL` (`aggregate.control_name_col`) matches it against another column.

BATCH_COLS
  `aggregate.batch_cols` · default `None`. Metadata columns that define a batch for alignment and TVN, usually `["plate", "well"]`.

TVN_BATCH_CORRECTION
  `aggregate.tvn_batch_correction` · default `True`. Fit {term}`TVN` per batch on that batch's controls. Set `False` when batches hold few control cells (it needs roughly ten per principal component per batch) or when only some batches contain controls.

SKIP_PERTURBATION_SCORE
  `aggregate.skip_perturbation_score` · default `False`. Skip {term}`perturbation scoring <Perturbation score>`; `PS_PROBABILITY_THRESHOLD` and `PS_PERCENTILE_THRESHOLD` set which cells it keeps.

AGG_METHOD
  `aggregate.agg_method` · default `None`. `"median"` (usual) or `"mean"`.

NUM_SIMS
  `aggregate.num_sims` · default `None`. Number of {term}`bootstrap <Bootstrap scoring>` simulations, for example `100000`. The bootstrap runs only for the cell classes and channel combos listed in `BOOTSTRAP_CELL_CLASS` and `BOOTSTRAP_CHANNEL_COMBO`.
```

## Cluster (`10_cluster.py`)

```{glossary}
MIN_CELL_CUTOFFS
  `cluster.min_cell_cutoffs` · default `None`. Minimum cells per perturbation for each cell class, for example `{"all": 5, "Interphase": 5, "Mitotic": 5}`. Lower it for rare classes.

SPECIES_ID (cluster)
  Not written; selects which {term}`benchmarks <Benchmarks>` are generated · default `"9606"` (human). NCBI taxonomy ID of the screen organism.

PHATE_DISTANCE_METRIC
  `cluster.phate_distance_metric` · default `None`. `"cosine"` (recommended) or `"euclidean"`.

PERTURBATION_AUC_THRESHOLD
  `cluster.perturbation_auc_threshold` · default `None`. Drop perturbations whose perturbation-score AUC is below this, for example `0.6`. Controls are always kept.

CONTROL_SCOPE, CONTROL_REFERENCE_GROUP
  `cluster.control_scope`, `cluster.control_reference_group` · defaults `"pooled"`, `None`. The {term}`control scope <Control scope>` for each point's distance to controls. Matters only when `GROUP_COLS` was set in aggregate.

CLUSTER_CONTROL_KEY
  `cluster.control_key`, written only when set · default `None` (inherit `aggregate.control_key`). Control perturbations for clustering, when the aggregate key names a treatment rather than control perturbations.

FINAL_LEIDEN_RESOLUTIONS
  `cluster.leiden_resolutions` · default `None`. The Leiden resolutions the pipeline clusters and benchmarks; each gets its own output directory.
```

## Analyze and MozzareLLM (`12_analyze.py`)

```{glossary}
CHANNEL_COMBO, CELL_CLASS, LEIDEN_RESOLUTION
  `mozzarellm.channel_combo`, `mozzarellm.cell_class`, `mozzarellm.leiden_resolution` · default `None`. The clustering to study in the notebook, to annotate by default, and to open the visualizer on. `COMPARTMENT_COMBO` adds the compartment combo when aggregation was split.

MOZZARELLM_MODEL
  `mozzarellm.model` · default `"claude-sonnet-5"`. The language model; its provider's API key goes in `analysis/.env`.

MOZZARELLM_SOURCE
  `mozzarellm.source` · default `"affinage_then_uniprot"`. The functional annotation given to the model for each gene; see [MozzareLLM](3.running_modules.md#mozzarellm).

MOZZARELLM_RUN_NAME
  `mozzarellm.run_name` · default `"run1"`. Output directory name under each clustering's `mozzarellm/`. Keep it to resume an interrupted run; change it for a fresh one.

MOZZARELLM_EXTRA_CLUSTERINGS
  Written to the combo table (`mozzarellm.combo_fp`) · default `[]`. Further clusterings to annotate, each a dict of any of `channel_combo`, `cell_class`, `leiden_resolution`, `compartment_combo`.

MOZZARELLM_REWRITE_CONTEXTS
  Not written · default `False`. Re-derive every screen context from `screen.yaml`, discarding hand edits.
```
