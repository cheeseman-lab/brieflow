## OME-Zarr Migration Plan for TIFF Outputs

**Goal:** Extend OME-Zarr support beyond preprocess so that key TIFF outputs across the SBS and phenotype modules can be exported as `.ome.zarr` stores (for visualization and scalable access), while keeping the existing TIFF-based workflow stable and backwards compatible.

---

## 1. Current State Overview

### 1.1 Existing OME-Zarr Support

- **Writer implementation**: `lib/preprocess/preprocess.py`
  - `write_multiscale_omezarr(image, output_dir, pixel_size, chunk_shape, coarsening_factor, max_levels)`
  - `nd2_to_omezarr(files, output_dir, ...)` (wraps `nd2_to_tiff` + `write_multiscale_omezarr`)
- **Snakemake rules**: `workflow/rules/preprocess.smk`
  - `convert_sbs_omezarr` – ND2 → OME-Zarr for SBS preprocess images.
  - `convert_phenotype_omezarr` – ND2 → OME-Zarr for phenotype preprocess images.
- **Targets**: `workflow/targets/preprocess.smk`
  - `PREPROCESS_OUTPUTS["convert_sbs_omezarr"]` and `["convert_phenotype_omezarr"]` write under `preprocess/omezarr/...`.
- **Metadata helpers**: `lib/shared/omezarr_utils.py`
  - Currently focused on **channel color and OMERO metadata patching** for existing `.ome.zarr` stores.

### 1.2 TIFF-Producing Outputs (Original Workflow)

Below are the main TIFF outputs that are *not* yet mirrored as OME-Zarr:

- **Preprocess module**
  - `workflow/targets/preprocess.smk`
    - `convert_sbs` → `preprocess/images/sbs/...__image.tiff`
    - `convert_phenotype` → `preprocess/images/phenotype/...__image.tiff`
    - `calculate_ic_sbs` → `preprocess/ic_fields/sbs/...__ic_field.tiff`
    - `calculate_ic_phenotype` → `preprocess/ic_fields/phenotype/...__ic_field.tiff`

- **SBS module**
  - `workflow/targets/sbs.smk` (`SBS_OUTPUTS`)
    - `align_sbs` → `sbs/images/...__aligned.tiff`
    - `log_filter` → `sbs/images/...__log_filtered.tiff`
    - `compute_standard_deviation` → `sbs/images/...__standard_deviation.tiff`
    - `find_peaks` → `sbs/images/...__peaks.tiff`
    - `max_filter` → `sbs/images/...__max_filtered.tiff`
    - `apply_ic_field_sbs` → `sbs/images/...__illumination_corrected.tiff`
    - `segment_sbs` → `sbs/images/...__nuclei.tiff` and `...__cells.tiff`
  - Scripts reading/writing TIFFs:
    - `workflow/scripts/sbs/align_cycles.py`
    - `workflow/scripts/sbs/log_filter.py`
    - `workflow/scripts/sbs/compute_standard_deviation.py`
    - `workflow/scripts/sbs/max_filter.py`
    - `workflow/scripts/sbs/find_peaks.py`
    - `workflow/scripts/sbs/apply_ic_field_sbs.py`
    - Downstream readers (e.g. `extract_bases.py`, `call_reads.py`) use `tifffile.imread`.

- **Phenotype module**
  - `workflow/targets/phenotype.smk` (`PHENOTYPE_OUTPUTS`)
    - `apply_ic_field_phenotype` → `phenotype/images/...__illumination_corrected.tiff`
    - `align_phenotype` → `phenotype/images/...__aligned.tiff`
    - `segment_phenotype` → `phenotype/images/...__nuclei.tiff`, `...__cells.tiff`
    - `identify_cytoplasm` → `phenotype/images/...__identified_cytoplasms.tiff`
  - Rules consuming these: `workflow/rules/phenotype.smk`
    - `align_phenotype`, `segment_phenotype`, `identify_cytoplasm`, `extract_phenotype_cp`, etc.
  - Scripts using TIFF IO:
    - `workflow/scripts/phenotype/align_phenotype.py`
    - `workflow/scripts/phenotype/apply_ic_field_phenotype.py`
    - `workflow/scripts/phenotype/identify_cytoplasm_cellpose.py`
    - `workflow/scripts/phenotype/extract_phenotype_cp_multichannel.py`
    - `workflow/scripts/shared/segment.py`, `segment_ph.py`, `extract_phenotype_minimal.py`

- **Other TIFF users**
  - QC / metrics:
    - `workflow/lib/shared/metrics.py` – counts `*.tiff` for sanity checks.
    - `workflow/lib/shared/eval_segmentation.py` – expects `_nuclei.tiff`, `_cells.tiff`, `__illumination_corrected.tiff`.
  - Montage/visualization:
    - `workflow/lib/aggregate/montage_utils.py` – reads TIFFs when generating overlays.
    - `workflow/scripts/aggregate/generate_montage.py` – writes TIFF overlays (plus PNGs).
    - `visualization/pages/4_Cluster_Analysis.py` – exposes `.tiff` overlays for download.

**Key point:** The computational pipeline today is tightly coupled to TIFF for intermediate processing, while OME-Zarr is only used for *preprocess* ND2 conversions.

---

## 2. Recommendation

- **Do not replace TIFF everywhere.**
  - Many scripts and libraries (`tifffile.imread/imwrite` calls across SBS/phenotype/aggregate) assume TIFF; rewriting them all to be OME-Zarr–native would be a **large, high-risk refactor**.
  - Some outputs (e.g. montages, overlay diagnostics) are relatively small and primarily used for export/download; zarr adds overhead without much benefit there.

- **Do extend OME-Zarr exports to cover all major image “checkpoints”.**
  - For large, information-rich images where interactive inspection is valuable, having OME-Zarr in addition to TIFF is **highly beneficial**:
    - Preprocess images (already supported).
    - SBS aligned / illumination-corrected images and segmentation masks.
    - Phenotype aligned / illumination-corrected images, segmentation masks, and cytoplasm masks.
  - This keeps the existing TIFF-based compute path stable while giving you a **uniform OME-Zarr view of the whole pipeline**.

- **Make OME-Zarr exports configuration-driven and optional.**
  - Add per-module config switches and stage lists (e.g. `sbs.omezarr_stages`) so you can choose:
    - Which stages to export as OME-Zarr.
    - Whether to keep or eventually drop certain TIFF outputs in storage.

**Conclusion:** It is beneficial to *add* OME-Zarr exports for essentially all important TIFF checkpoints, but not to fully replace TIFF as the working format right now. The plan below assumes: **TIFF remains the internal compute format; OME-Zarr becomes an optional, standardized export layer.**

---

## 3. Phase 1 – Introduce a Generic TIFF → OME-Zarr Export Helper

### 3.1 Implement `export_tiff_to_omezarr` in `lib/shared/omezarr_utils.py`

- **Add a utility** that reuses the existing writer in `lib/preprocess/preprocess.py`:
  - **Signature (high-level):**
    - `export_tiff_to_omezarr(tiff_path, output_dir, pixel_size=(1.0, 1.0), chunk_shape=(1, 512, 512), coarsening_factor=2, max_levels=None, channel_labels=None, label_metadata=None) -> Path`
  - **Behavior:**
    - Read TIFF via `tifffile.imread`.
    - Normalize to CYX:
      - If 2D: add channel axis.
      - If other shapes (e.g. YXC), permute as needed.
    - Call `write_multiscale_omezarr`.
    - Optionally:
      - Patch `omero.channels[*].label` using `channel_labels`.
      - Set any label-specific metadata if `label_metadata` is provided.

### 3.2 Add simple channel-color / label helpers

- Reuse or extend `ensure_omero_channel_colors` to:
  - Ensure `omero.channels` exists and has default colors for each channel.
  - Optionally set labels: nuclei, cells, cytoplasm, etc.

### 3.3 Unit tests for the helper

- **New tests** (e.g. `tests/unit/test_export_tiff_to_omezarr.py`):
  - Roundtrip small synthetic TIFFs (2D + multi-channel) → OME-Zarr.
  - Ensure:
    - `scale0` data matches original.
    - `.zattrs` contains multiscales + omero.
    - Optional `channel_labels` are applied when provided.

---

## 4. Phase 2 – SBS Module: OME-Zarr Exports for TIFF Outputs

### 4.1 Identify SBS stages worth exporting

**High-value for visualization / QC:**

- `align_sbs` – aligned SBS images.
- `apply_ic_field_sbs` – illumination-corrected SBS images.
- `segment_sbs` – nuclei + cell label images.

**Lower-value / optional (debug-focused):**

- `log_filter` – log-filtered images.
- `compute_standard_deviation` – per-pixel stddev across cycles.
- `find_peaks` – peak maps.
- `max_filter` – max-filtered images.

### 4.2 Add new SBS OME-Zarr targets (`workflow/targets/sbs.smk`)

- **Extend `SBS_OUTPUTS`** with OME-Zarr directories, parallel to existing TIFF paths, e.g.:
  - `align_sbs_omezarr` → `sbs/omezarr/aligned/...__aligned.ome.zarr`
  - `apply_ic_field_sbs_omezarr` → `sbs/omezarr/illumination_corrected/...__illumination_corrected.ome.zarr`
  - `segment_sbs_omezarr` → `sbs/omezarr/segmentation/...__segmentation.ome.zarr`
  - Optionally: `log_filter_omezarr`, `max_filter_omezarr`, etc.
- **Add `SBS_OUTPUT_MAPPINGS` entries** with `directory` mapping for OME-Zarr outputs (so Snakemake treats them as directories).
- **Add `SBS_TARGETS_OMEZARR`** (list of stage-specific targets) and keep them separate from `SBS_TARGETS_ALL` for configurability.

### 4.3 Add SBS export rules (`workflow/rules/sbs.smk`)

- **New rules** that take existing TIFF outputs as input and emit OME-Zarr:

  - `export_align_sbs_omezarr`:
    - `input`: `SBS_OUTPUTS["align_sbs"]`
    - `output`: `SBS_OUTPUTS_MAPPED["align_sbs_omezarr"]`
    - `params`: chunk shape, coarsening, max levels, channel names.
    - `script`: `../scripts/sbs/export_align_sbs_omezarr.py`

  - `export_apply_ic_field_sbs_omezarr`:
    - `input`: `SBS_OUTPUTS["apply_ic_field_sbs"]`
    - `output`: `SBS_OUTPUTS_MAPPED["apply_ic_field_sbs_omezarr"]`
    - Same params pattern as above.
    - `script`: `../scripts/sbs/export_ic_corrected_sbs_omezarr.py`

  - `export_segment_sbs_omezarr`:
    - `input`: two TIFF masks from `segment_sbs` (nuclei, cells).
    - `output`: `SBS_OUTPUTS_MAPPED["segment_sbs_omezarr"]`
    - `script`: `../scripts/sbs/export_segment_sbs_omezarr.py`
      - Stack masks into channels and treat as label image.

### 4.4 SBS export scripts (`workflow/scripts/sbs/*.py`)

- **Pattern for image-like exports** (aligned, IC-corrected, etc.):
  - Read TIFF with `tifffile.imread`.
  - Call `export_tiff_to_omezarr` with:
    - `pixel_size` (can be set to `[1.0, 1.0]` or taken from preprocess metadata if desired).
    - `chunk_shape`, `coarsening_factor`, `max_levels` from `snakemake.params`.
    - `channel_labels` from `config["sbs"]["channel_names"]`.

- **Pattern for segmentation exports**:
  - Read nuclei and cell masks.
  - Stack into a 2-channel array.
  - Call `export_tiff_to_omezarr` with:
    - `channel_labels=["Nuclei", "Cells"]`.
    - Optional `label_metadata` block for image-label semantics (if you want to adopt the image-label convention later).

### 4.5 Configuration toggles (`config.yml`)

- Under `sbs:` section, add:

```yaml
sbs:
  omezarr_export: true          # master toggle for SBS OME-Zarr exports
  omezarr_chunk_shape: [1, 512, 512]
  omezarr_coarsening_factor: 2
  omezarr_max_levels: 5
  # Which SBS stages to export as OME-Zarr
  omezarr_stages:
    - align_sbs
    - apply_ic_field_sbs
    - segment_sbs
    # optional:
    # - log_filter
    # - compute_standard_deviation
    # - find_peaks
    # - max_filter
```

- Wire `SBS_TARGETS_OMEZARR` into `SBS_TARGETS_ALL` conditionally in `workflow/rules/sbs.smk` or a central place (e.g. `Snakefile`).

---

## 5. Phase 3 – Phenotype Module: OME-Zarr Exports

### 5.1 Phenotype stages to export

**High-value:**

- `apply_ic_field_phenotype` → IC-corrected images.
- `align_phenotype` → aligned phenotype images (used by CP).
- `segment_phenotype` → nuclei and cell masks.
- `identify_cytoplasm` → cytoplasm masks.

### 5.2 New phenotype OME-Zarr targets (`workflow/targets/phenotype.smk`)

- Extend `PHENOTYPE_OUTPUTS` with:
  - `apply_ic_field_phenotype_omezarr` → `phenotype/omezarr/illumination_corrected/...__illumination_corrected.ome.zarr`
  - `align_phenotype_omezarr` → `phenotype/omezarr/aligned/...__aligned.ome.zarr`
  - `segment_phenotype_omezarr` → `phenotype/omezarr/segmentation/...__segmentation.ome.zarr`
  - `identify_cytoplasm_omezarr` → `phenotype/omezarr/cytoplasm/...__identified_cytoplasms.ome.zarr`
- Update `PHENOTYPE_OUTPUT_MAPPINGS` to mark these as `directory`-mapped outputs.
- Add a `PHENOTYPE_TARGETS_OMEZARR` list for stage-specific OME-Zarr targets.

### 5.3 Phenotype export rules (`workflow/rules/phenotype.smk`)

- Add rules similar to SBS:

  - `export_apply_ic_field_phenotype_omezarr`
  - `export_align_phenotype_omezarr`
  - `export_segment_phenotype_omezarr`
  - `export_identify_cytoplasm_omezarr`

- Each rule:
  - Takes the corresponding TIFF output as `input`.
  - Emits the new OME-Zarr directory as `output`.
  - Uses `chunk_shape`, `coarsening_factor`, `max_levels` from `config["phenotype"]`.
  - Provides `channel_labels` consistent with `config["phenotype"]["channel_names"]` or fixed labels for masks (`["Nuclei", "Cells"]`, `["Cytoplasm"]`).

### 5.4 Phenotype export scripts (`workflow/scripts/phenotype/*.py`)

- Implement `export_*_phenotype_omezarr.py` scripts that:
  - Read the relevant TIFF(s) with `tifffile.imread`.
  - Preprocess into CYX if needed.
  - Call `export_tiff_to_omezarr`.

### 5.5 Phenotype config toggles

- Under `phenotype:` in `config.yml`, add:

```yaml
phenotype:
  omezarr_export: true
  omezarr_chunk_shape: [1, 512, 512]
  omezarr_coarsening_factor: 2
  omezarr_max_levels: 5
  omezarr_stages:
    - apply_ic_field_phenotype
    - align_phenotype
    - segment_phenotype
    - identify_cytoplasm
```

- As with SBS, merge `PHENOTYPE_TARGETS_OMEZARR` into `PHENOTYPE_TARGETS_ALL` conditionally.

---

## 6. Phase 4 – Optional Exports for Other TIFF Artifacts

These are **nice-to-have** and can be added later, using the same pattern with `export_tiff_to_omezarr`:

- **Illumination correction fields**:
  - `preprocess/ic_fields/sbs` and `preprocess/ic_fields/phenotype`.
  - Use a label/technical channel naming convention (e.g. `"IC field DAPI"`, `"IC field cytoplasm"`).

- **Paramsearch / QC outputs**:
  - TIFFs referenced by `eval_segmentation.py` for segmentation paramsearch.
  - These are debug-focused; OME-Zarr can help when visually scanning many conditions.

- **Aggregate montages / overlays**:
  - Final montages and overlays from `aggregate` are smaller images; you can:
    - Keep TIFF/PNG as the main export.
    - Optionally add OME-Zarr overlays if browser-based viewers or cloud storage are planned.

Implementation pattern mirrors Phases 2–3: add targets → rules → export scripts, gated by config flags (e.g. `aggregate.omezarr_export_overlays`).

---

## 7. Phase 5 – Configuration, Backwards Compatibility, and Storage Strategy

### 7.1 Keep TIFF as the compute and default visualization format (for now)

- All existing rules and scripts should continue to:
  - **Read** from TIFF.
  - **Write** their current TIFF outputs.
- OME-Zarr exports run as **additional rules** driven by configuration, not replacements.

### 7.2 Global and per-module switches

- Introduce a global flag and per-module overrides, e.g.:

```yaml
all:
  omezarr_export_default: false   # safe default for existing users

sbs:
  omezarr_export: true            # overrides global
  omezarr_stages: [align_sbs, apply_ic_field_sbs, segment_sbs]

phenotype:
  omezarr_export: true
  omezarr_stages: [align_phenotype, segment_phenotype, identify_cytoplasm]
```

- `Snakefile` (or module-level setup) uses these flags to decide whether to:
  - Add OME-Zarr targets to `all_*` aggregates.
  - Expose additional convenience rules, e.g. `all_sbs_omezarr`, `all_phenotype_omezarr`.

### 7.3 Storage & cleanup

- As OME-Zarr is added, storage may increase:
  - Per-image overhead due to multiscale levels.
  - Additional directory depth and metadata.
- Mitigation options:
  - Allow **per-stage** toggles (e.g. only export `aligned` and `segment_*` stages).
  - Add cleanup scripts / rules (e.g. Snakemake `cleanup_omezarr_*`) to remove intermediate OME-Zarr directories if only final ones are needed long term.

---

## 8. Phase 6 – Testing Strategy

### 8.1 Unit tests

- **For `export_tiff_to_omezarr`**:
  - Roundtrip synthetic TIFFs with shapes:
    - `(Y, X)` and `(C, Y, X)` for both integer and float dtypes.
  - Verify:
    - `scale0` matches the original array.
    - `multiscales` length matches the intended number of pyramid levels.
    - `omero.channels` have the correct labels when provided.

### 8.2 Integration tests: SBS and Phenotype

- Extend existing integration tests or add new ones:
  - **SBS:**
    - Run `all_sbs` with `sbs.omezarr_export: true` on `small_test_analysis`.
    - Assert `.ome.zarr` directories exist for selected stages.
    - Check `scale0.shape` matches the corresponding TIFF.
  - **Phenotype:**
    - Run `all_phenotype` with `phenotype.omezarr_export: true`.
    - Assert `.ome.zarr` for `align_phenotype`, `segment_phenotype`, etc.
    - Verify segmentation channels encode integer labels correctly.

### 8.3 Manual visualization checks

- Use Napari / Vizarr to confirm:
  - Correct channel ordering and labeling.
  - Reasonable default contrast (stretch based on dtype range).
  - For segmentation OME-Zarrs, verify that labels align with original TIFF masks.

---

## 9. Longer-Term (Optional) Work: Making the Pipeline Format-Aware

Once OME-Zarr exports are stable and widely used, consider a **second-stage project** to make the pipeline capable of directly consuming OME-Zarr:

- Introduce a generic `read_image(path)` helper in `lib/shared` that:
  - Auto-detects TIFF vs `.ome.zarr` (dir with `.zattrs`).
  - Returns a CYX `numpy` array (for now).
- Gradually refactor scripts that currently import `tifffile.imread` directly to use `read_image`.
- Add config knobs like `sbs.input_format` / `phenotype.input_format` to switch between TIFF and OME-Zarr for computational steps.


