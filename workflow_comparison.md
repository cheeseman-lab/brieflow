# Workflow Comparison Report

This report summarizes the differences between the legacy `brieflow` workflow and the updated `@workflow` codebase.

## 1. Structure & Configuration

### `workflow/Snakefile`
*   **Imports:** Legacy imports `get_metadata_wildcard_combos` from `lib.preprocess.file_utils`. New imports `initialize_segment_sbs_paramsearch`, `initialize_segment_phenotype_paramsearch` from `lib.shared.initialize_paramsearch`.
*   **Metadata:** Legacy handles `SBS_METADATA_SAMPLES_DF_FP` and `PHENOTYPE_METADATA_SAMPLES_DF_FP` for external metadata files. This logic is **removed** in the new workflow.
*   **Parameter Search:** New workflow adds logic for `segment_sbs_paramsearch` and `segment_phenotype_paramsearch` modes.
*   **Testing:** New workflow includes commented-out code for restricting wells/tiles for testing.

## 2. Rules & Targets

### `aggregate.smk`
*   **Bootstrap:** Legacy includes rules for bootstrap statistical testing (`bootstrap_construct`, `bootstrap_gene`, `combine_bootstrap`). These are **removed** in the new workflow.
*   **Inputs:** `split_datasets` now uses `ancient(MERGE_OUTPUTS["final_merge"])`.
*   **Montage:** `generate_montage` uses `channel=config["phenotype"]["channel_names"]` in `expand`.

### `cluster.smk`
*   **Filtering:** Legacy `phate_leiden_clustering` supports `perturbation_auc_threshold` filtering. This is **removed** in the new workflow.
*   **Inputs:** `clean_aggregate` input updated to `ancient(AGGREGATE_OUTPUTS["aggregate"])`.
*   **Benchmarks:** `benchmark_clusters` rule now explicitly uses `corum_group_benchmark_fp` and `kegg_group_benchmark_fp` from config.

### `merge.smk`
*   **Approach:** Legacy supports "fast" (tile-based) and "stitch" (well-based) merge approaches. New workflow **removes the "stitch" approach** and related rules (`estimate_stitch`, `stitch_phenotype`, `stitch_sbs`, `stitch_alignment`, `stitch_merge`), retaining only the fast/tile-based alignment.
*   **Fast Alignment:** `fast_alignment` params updated (e.g., `sbs_metadata_filters`). Coordinate alignment transformations (`flip_x`, etc.) are removed.
*   **Evaluation:** `eval_merge` outputs updated to include `all_cells_by_channel_min` and `cells_with_channel_min_0` plots.

### `phenotype.smk`
*   **Segmentation:** `segment_phenotype` now uses `get_segmentation_params` helper.
*   **Parameters:** `extract_phenotype_cp` uses `foci_channel` (value) instead of `foci_channel_index`.

### `preprocess.smk`
*   **OME-Zarr:** New workflow adds support for OME-Zarr conversion (`convert_sbs_omezarr`, `convert_phenotype_omezarr`).
*   **Scripts:** `image_to_tiff.py` replaced by `nd2_to_tiff.py` (and `nd2_to_omezarr.py`).
*   **Metadata:** Metadata extraction logic simplified; `extract_tile_metadata.py` replaces generic `extract_metadata.py`. `combine_metadata` rules now use generic `combine_dfs.py`.

### `sbs.smk`
*   **Methods:** `call_reads` rule now uses `call_reads_method`. `find_peaks` logic updated for `spot_detection_method`.
*   **IC Field:** `apply_ic_field_sbs` uses `segmentation_cycle_index` instead of named cycle params.
*   **Cell Calling:** `call_cells` params explictly listed instead of using `get_call_cells_params`.

## 3. Library (Lib) Changes

### `lib/aggregate`
*   **`aggregate.py`**: Removed `ps_probability_threshold` and `ps_percentile_threshold` filtering. Removed `perturbation_auc` and `aggregated_perturbation_score` calculation.
*   **`align.py`**: Removed `median_absolute_deviation`. Removed `perturbation_score` calculation logic.
*   **Removed Files:** `bootstrap.py`, `perturbation_score.py`.

### `lib/merge`
*   **`deduplicate_merge.py`**: Removed `analyze_distance_distribution` and `approach` (stitch/fast) logic.
*   **`eval_merge.py`**: Removed complex visualization functions (`display_matched_and_unmatched_cells_for_site`, `create_enhanced_match_visualization`, `run_well_alignment_qc`).
*   **Removed Files:** `estimate_stitch.py`, `eval_stitch.py`, `stitch_alignment.py`, `stitch_merge.py`, `stitch.py`, `merge_utils.py`.

### `lib/phenotype`
*   **`align_channels.py`**: Removed `visualize_phenotype_alignment`.
*   **`constants.py`**: `DEFAULT_METADATA_COLS` now uses `sgRNA_0` instead of `cell_barcode_0`.

### `lib/preprocess`
*   **`file_utils.py`**: Removed `z` parameter from `get_sample_fps`. Removed metadata extraction helpers (`get_metadata_wildcard_combos`, etc.).
*   **`preprocess.py`**: Major refactor. Removed `extract_metadata` (generic), `convert_to_array`. Added `nd2_to_tiff`, `nd2_to_omezarr`, `extract_tile_metadata`.

### `lib/sbs`
*   **`align_cycles.py`**: Removed `visualize_sbs_alignment`.
*   **`call_cells.py`**: Simplified `call_cells` signature.
*   **Removed Files:** `automated_parameter_search.py`, `barcode_cycle_utils.py`, `standardize_barcode_design.py` (logic moved/removed?).

### `lib/shared`
*   **`io.py`**: **New file** providing unified `read_image` and `save_image` supporting both TIFF and OME-Zarr.
*   **`omezarr_io.py`, `omezarr_utils.py`**: **New files** for OME-Zarr support.
*   **`metrics.py`**: **New file** for generating run metrics.
*   **`rule_utils.py`**: Updated `get_alignment_params` and `get_segmentation_params`. Removed bootstrap and call_cells helpers.
*   **`segment_cellpose.py`**: Simplified. Removed `helper_index` support. Removed Cellpose version compatibility checks (assumes compatible version).
*   **`segmentation_utils.py`**: **Removed**. Functions likely moved/inlined.

## 4. Scripts

*   **`scripts/preprocess`**: `image_to_tiff.py` replaced by `nd2_to_tiff.py`. `extract_metadata.py` updated to use `extract_tile_metadata`.
*   **`scripts/sbs`**: `align_cycles.py`, `apply_ic_field_sbs.py`, `compute_standard_deviation.py`, `find_peaks.py`, `log_filter.py`, `max_filter.py` updated to use `lib.shared.io` (`read_image`/`save_image`) and handle pixel size.
*   **`scripts/shared`**: `segment.py` updated to use `read_image`/`save_image` and new param structure.

## Summary
The new workflow represents a significant refactoring with the following key themes:
1.  **OME-Zarr Support:** Added native support for reading/writing OME-Zarr images alongside TIFFs.
2.  **Simplified Merge:** Removed the complex "stitch" based merge approach, relying solely on the faster tile-based alignment.
3.  **Removed Features:** Bootstrap analysis, perturbation scoring, and some advanced visualization/QC tools have been removed.
4.  **Refactored Preprocessing:** Simplified metadata extraction and image conversion logic, removing support for external metadata files.
5.  **Unified I/O:** Introduction of `lib.shared.io` to standardize image access.
