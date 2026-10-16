# Multiround nuclear bodies and refined 53BP1 boundaries

This opt-in phenotype extension detects subnuclear objects, associates them with their parent nuclei, and measures registered marker intensities.
It is an application-specific MN-body workflow with configurable thresholds; it currently expects five imaging rounds and the seven marker roles listed below.
It does not change ordinary Brieflow runs when `phenotype.mn_bodies.enabled` is absent or false.

## Workflow and configuration

Use the usual nuclear segmentation with `segment_cells: false` and, for Cellpose-SAM, `cellpose_model: cpsam` in a verified Cellpose 4 environment.
Keep `align: false` and remove custom channel offsets because this extension performs its own DAPI registration.
Supply a complete `phenotype.mn_bodies` configuration with `channel_names`, `rounds`, `objects`, `nucleoli`, `reporter`, `consensus`, `H3K9me2_artifacts`, and `measurement` settings.
The {download}`example configuration <mn_bodies.example.yml>` contains a full annotated configuration, calibrated to one pilot's pixel size and intensity scale rather than universal defaults.

1. `register_mn_rounds` estimates DAPI translations and applies each round's transform to all its markers, retaining validity and source-saturation information.
2. `export_mn_registered_image` supplies the aligned multichannel image to standard nuclear feature extraction.
3. `segment_mn_components` detects 53BP1, original and later-round RNP, FLAG reporters, and nucleolar proxies inside parent nuclei.
4. `refine_mn_components` optionally trims existing 53BP1 footprints, preserving original object IDs and original masks.
5. `define_mn_bodies` recomputes RNP contacts and consensus/rescue decisions using the refined footprints, then assigns reporter status.
6. `measure_mn_bodies` measures the seven markers and depleted-parent baselines; `merge_mn_phenotype` joins a one-row-per-parent summary to ordinary phenotype features.

The current marker roles are 53BP1 (`BP1`), `H2AK119ub`, `H3K27ac`, `H3K9me2`, `PolII_Ser5Ph`, `PolII_Ser2Ph`, and nascent transcription (`Nascent`).
RNP and FLAG are also used for object detection and association.
Round/channel mappings are explicit and must match the actual CYX image stack.
The same tile and parent label remain the link to subsequent SBS assignments.

## Peak-relative boundary refinement

Add this block under `phenotype.mn_bodies` to reproduce the reviewed 30% setting:

```yaml
bp1_boundary:
  enabled: true
  method: relative_contrast
  threshold: 0.30
  smoothing_sigma: 0.7
  ring_inner: 2
  ring_outer: 8
  minimum_background_pixels: 30
```

Omitting the block or setting `enabled: false` preserves the incoming 53BP1 masks.
An enabled block requires an explicit threshold; 30% is a pilot choice, not a universal default.
All distances and areas are in reference-image pixels.

For each existing focus, smooth the unenhanced 53BP1 channel within its own parent nucleus using a normalized Gaussian filter.
Let `P90` be the 90th percentile of this smoothed signal inside the original focus.
Let `B` be the median unenhanced 53BP1 signal in a local annulus, intersected with the original depleted parent nucleus.
The original depleted mask excludes original 53BP1, both RNP masks, nucleolar proxies and accepted reporters and is held fixed while refining.
If the annulus contains fewer than 30 available pixels, use the entire available depleted parent as the background reference.

Keep candidate pixels satisfying:

```text
smoothed_53BP1 >= B + 0.30 * max(P90 - B, 0)
```

Retain the connected component containing the original smoothed maximum, fill enclosed holes within the original mask, and preserve its label.
The result is a subset of the original footprint; it cannot discover a missed focus or enlarge one.
Thirty percent describes signal contrast, not a fraction of area to remove.
Measurements always use unenhanced intensities, never the smoothed image used for boundaries.
If no background pixels are available, retain the original focus and explicitly flag `background_unavailable`; do not silently invent a threshold.
Downstream baseline QC remains independent.

Reporter association labels and epigenetic enrichment do not enter the cutoff formula.
However, reporter and nucleolar exclusion masks can influence the available background pixels; this workflow is not fully independent of those masks.

## MN calls, reporter status, and measurements

Define A and C as refined 53BP1 objects contacting original and later-round RNP, respectively, within the same nucleus.
Contact includes pixel overlap or immediate edge/corner adjacency; it does not bridge a gap.
The pilot uses:

```text
(A AND C) OR ((A OR C) AND BP1 enrichment >= 5 AND refined area >= 70 px)
```

The area cutoff applies to the refined area, so some previously rescued objects can be lost after trimming.
Recompute contacts, rescue enrichment, reporter overlap and all marker baselines after refinement; changing only the displayed contour would be inconsistent.
An MN is reporter-associated only when more than 20% of one individual accepted reporter lies inside that individual MN.
The denominator is reporter area, not MN area; separate partial reporters are not summed.
Unavailable reporter coverage is distinguished from reporter-negative.

The rescue baseline excludes refined 53BP1, both RNP masks and nucleolar proxies.
The measurement baseline additionally excludes accepted reporters.
The boundary background instead uses the original masks, avoiding circular threshold updates.
Nucleolar proxies subtract from baselines but never veto MN bodies.
For H3K9me2, bright artifacts are removed from the baseline and overlapping MN bodies fail QC for this marker only.

Canonical enrichment is mean marker signal in the MN divided by mean signal in its own depleted parent nucleus.
Additional outputs include integrated signal and intensity evenness.
Pol II level is `(x+y)/2`, and balance is `log2(x/y)`, for positive normalized Ser5Ph and Ser2Ph values.
These are relative enrichment indices, not calibrated molecular amounts or biochemical rates.

## Outputs and implementation

| Output suffix | Purpose |
|---|---|
| `mn_components.npz` | Original segmentation masks, retained unchanged |
| `mn_refined_components.npz` | Refined `bp1`, original `bp1_untrimmed`, and the other component masks |
| `mn_bp1_boundaries.tsv` | Per-object areas, cutoff, background, fallback flags and signal retention |
| `mn_boundary_provenance.json` | Settings, source hashes, implementation hashes and fallback counts |
| `mn_derived.npz` | MN masks, reporter partitions, and depleted-parent baselines |
| `mn_decisions.tsv` | A/C contacts, rescue tests and inclusion/exclusion reasons |
| `mn_features.tsv` / well-level parquet | Object-level measurements and marker-specific QC |
| `mn_summary.tsv` | Parent-level counts and summary measurements, including observed zeros |
| `mn_phenotype_with_mn.tsv` | Validated parent-summary join for the ordinary phenotype/SBS route |

The boundary algorithm lives in `workflow/lib/phenotype/bp1_boundary_refinement.py`.
Array orchestration and file adapters live in `workflow/lib/phenotype/mn_pipeline.py`.
Snakemake dependencies and output naming live in `workflow/rules/mn_bodies.smk` and `workflow/targets/mn_bodies.smk`.
`workflow/scripts/phenotype/run_mn_stage.py` is a thin adapter.
Rule parameters record the settings and algorithm fingerprints.
Changing only `bp1_boundary` settings restarts refinement and its dependants, reusing registration and component detection.
Library changes invalidate the recorded implementation fingerprints and can rerun earlier stages.

Object keys are `(plate, well, tile, mn_id)`; the parent join uses `parent_nucleus_id` against the normal nucleus `label`.
Tile-overlap deduplication and later genotype joins are separate downstream operations.
Historical reports retain their original masks and measurements; they do not become updated merely by changing a configuration file.

## Validation scope

Portable tests exercise nested footprints, label preservation, disabled settings, missing background, reporter-observation domains, strict overlaps, empty populations, baseline QC and refined-area rescue behavior.
A separate six-field pilot regression compares native masks and measurements with the reviewed threshold sweep.
That regression reuses previously reviewed Cellpose-SAM nuclei and preprocessed inputs; it does not validate fresh neural-network inference, a new microscope, full-screen processing or SBS matching.
The base dependency file at the development starting revision pins Cellpose 3.1.0, so a clean installation must resolve the Cellpose 4 requirement before running `cpsam`.
