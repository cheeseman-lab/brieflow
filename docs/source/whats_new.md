# What's New

Brieflow 1.5.1 is the release these docs describe.
Each release's full list of changes, with the pull request behind each, is in its [release notes](https://github.com/cheeseman-lab/brieflow/releases).

## 1.5.1

A maintenance release on top of 1.5.0; the [1.5.1 release notes](https://github.com/cheeseman-lab/brieflow/releases/tag/v1.5.1) list every change.

### Main changes

- **OME-Zarr by default.** A config without `all.image_format` now writes {term}`OME-Zarr`, as the preprocess notebook already did. TIFF and zarr runs both write to `brieflow_output/`. See [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
- **Cellpose 3 and 4 together.** Both install with brieflow, and `CELLPOSE_MODEL` picks the version: `cpsam` runs Cellpose 4, the other models Cellpose 3.1.0. No separate Cellpose 4 install; installing brieflow needs `git` and network access.
- **Parquet per-tile tables.** The per-tile tables later steps combine (SBS `reads`, `cells`, `sbs_info`; phenotype `phenotype_cp`) are parquet under `parquets/`, and the final merge streams through polars.
- **Metadata once per well.** Preprocess extracts image-header metadata in one job per well (and cycle or round) instead of one per tile.
- **Finished runs stay finished.** Running a finished screen again has nothing to do. Before, zarr runs re-ran segmentation and everything after it.
- **Cluster waits for the bootstrap.** For a combo with a bootstrap, cluster always runs after it, so `cluster.h5ad` always carries the bootstrap p-values.
- **StarDist in every install.** StarDist and a CPU build of TensorFlow install with brieflow.
- **Figures on white.** QC, cluster and classifier plots are saved on a white background; `TRANSPARENT_FIGURES` in the preprocess notebook turns transparent backgrounds back on. Plate heatmaps are laid out by plate row and column.
- **Optional within-cycle SBS alignment.** `USE_ALIGN_WITHIN_CYCLE` in the SBS notebook turns off the registration of each cycle's base channels, for data where it shifts channels by many pixels.
- **Tied dye states in combinatorial calling.** The `frac` caller can call a cycle whose dye state is equally near two bases `N`, and cell calling can resolve or skip those cycles. Off unless set.
- **Integer metadata stays integer** in the aggregate parquets, instead of becoming float32.

### Upgrading from 1.5.0

Start screens fresh on 1.5.1; outputs from earlier versions are not reused.

## 1.5.0

Brieflow 1.5.0 moved the pipeline to OME-Zarr and the marimo notebooks.
The full list of changes, with the pull request behind each, is in the [1.5.0 release notes](https://github.com/cheeseman-lab/brieflow/releases/tag/v1.5.0).

### Main changes

- **marimo notebooks.** brieflow-analysis configures each phase in a marimo notebook, `0_preprocess.py` … `7_analyze.py`, with the values you set gathered in {term}`parameter cells <Parameter cell>`. `flow.sh <module>` replaces the numbered run scripts. See [Working with the Notebooks](notebook_editor_setup.md).
- **OME-Zarr output.** `all.image_format: zarr` writes {term}`OME-Zarr` plate stores; `tiff` keeps one file per tile. See [Output formats](3.running_modules.md#output-formats-tiff-and-zarr).
- **Merge.** The `positions` approach matches cells from stage positions; the `stitch` approach is removed.
- **Screen description.** `analysis/screen.yaml` describes the screen, and an optional {term}`per-well annotations <Per-well annotations>` table selects the wells to process and the groups to aggregate.
- **Raw data layout.** Inputs are expected one directory per SBS cycle and per phenotype round; see [Before You Screen](1.before_you_screen.md).
- **Aggregate and cluster.** Grouped aggregation, bootstrap control scopes, and a bootstrap that keeps no null distributions on disk.
- **Python 3.12**, and the small test run in both output formats as the pipeline's test.

### Upgrading from 1.4.x

- Start a new screen from the brieflow-analysis template and follow [Installation and Analysis Setup](2.installation_analysis_setup.md).
- An existing screen keeps running the brieflow commit its submodule pins. To move it to 1.5, update the submodule, rebuild the environment, and configure each phase again in the marimo notebooks: the config keys and defaults changed, so a 1.4 `config.yml` is not reused as is.
- The release notes list every renamed or removed config key.
